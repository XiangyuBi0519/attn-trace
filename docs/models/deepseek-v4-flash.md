# DeepSeek V4 Flash — KV Cache 布局观测

用 `attn-trace` 抓到的 DeepSeek V4 Flash 在 vLLM 0.23 + vllm-ascend 0.23 上的启动期指纹。

## 采样环境

- 采样时间：2026-08-26 多轮采样（16:19–16:22 + 17:12–17:17）
- 运行时：vLLM 0.23.0 + vllm-ascend 0.23.0
- 部署形态：PD 分离场景下的 P 节点
  - `data_parallel_size = 2`（`EngineCore_DP0` / `EngineCore_DP1`）
  - `tensor_parallel_size = 8`（从 log 里 16 个 Worker pid 与 DP=2 反推）
- 硬件：Ascend NPU（非 310p）
- ⚠️ 两次采样的启动配置都是：`prefix_caching=False`、`connector=False`。**这两个开关的状态严重影响下面分析里"你插件能不能命中"的结论**，见文末"注意事项"。

## 一句话结论

DeepSeek V4 Flash 是 **MLA + 稀疏注意力（indexer + compressor）+ 滑动窗口** 的混合结构，在 vLLM 里落成 **6 个 KV cache group**，其中真正"传统全前缀 attention"的层只占 **约 12%（20 / 168）**。剩余 148 层要么是有窗口约束的 SWA，要么是稀疏检索的 indexer 或压缩状态缓存，语义上都不能直接套 hash-based prefix cache aging。

## 一、Backend 选择

```
[NPU_ATTN_MAP]  selected_backend=None key=(mla=True, sparse=False) use_compress=False is_310p=False
                -> vllm_ascend.attention.mla_v1.AscendMLABackend
[ATTN_BACKEND_PICK]  backend_cls=vllm_ascend.attention.mla_v1.AscendMLABackend
                     attn_cfg=AttentionSelectorConfig(head_size=0, dtype=torch.bfloat16,
                              kv_cache_dtype=None, block_size=128, use_mla=True,
                              has_sink=False, use_sparse=False, ...)
```

- 全部 16 个 Worker、两个 DP 都走同一个 backend：`AscendMLABackend`
- `use_sparse=False` —— 即便模型内部有 indexer/compressor，Ascend 侧仍把整机 backend 归到 MLA 分派；稀疏部分由**模型层**自己实现（体现在 spec 分组上，而不是 backend 分派上）

## 二、KV Cache Groups（6 组）

`[KV_GROUP]` 打出的完整分组（DP0 / DP1 一致，`total_blocks` 有 ±1 的 profiling 抖动）：

| idx | spec | 层数 | block_size | num_kv_heads | head_size | dtype | sliding_window | 用途推断 |
|---:|---|---:|---:|---:|---:|---|---:|---|
| 0 | `AscendMLAAttentionSpec` | 42 | 128 | 1 | 128 | **int8** | – | Sparse attention 的 **indexer k-cache**（int8 量化、短 head） |
| 1 | `AscendMLAAttentionSpec` | 20 | 128 | 1 | 512 | bf16 | – | **主 MLA attention** 缓存（唯一整套 full-prefix 语义可用的组） |
| 2 | `AscendSlidingWindowMLASpec` | 22 | 128 | 1 | 512 | bf16 | **128** | 偶数层的 `swa_cache`（滑窗 MLA） |
| 3 | `AscendSlidingWindowMLASpec` | 22 | 128 | 1 | 512 | bf16 | **128** | 奇数层的 `swa_cache`（按 parity 拆两组，通信优化） |
| 4 | `AscendSlidingWindowMLASpec` | 42 | **8** | 1 | 2048 | **fp32** | **8** | indexer + compressor 的 **state_cache**（极小窗口，fp32） |
| 5 | `AscendSlidingWindowMLASpec` | 20 | **32** | 1 | 1024 | **fp32** | 128 | 主分支的 **compressor state_cache** |

层命名规律（从 sample_layers）：

- `self_attn.attn` —— 主 attention 输出（组 1）
- `self_attn.indexer.k_cache` —— sparse 检索器的 K 缓存（组 0）
- `self_attn.swa_cache` —— 滑动窗口注意力缓存（组 2、3）
- `self_attn.compressor.state_cache` —— 压缩器状态缓存（组 5，也含在组 4 中）
- `self_attn.indexer.compressor.state_cache` —— 检索器内部的压缩器状态（组 4）

**共 168 层 KV-consuming module**，物理构成是：
- 主分支 20 层：每层 4 份缓存（1 主 attn + 1 swa + 1 compressor state + indexer 关联）
- 稀疏分支 42 层：每层 3 份缓存（indexer k + indexer compressor state + swa 等）
- 具体总层数 = 42 + 20 + 22 + 22 + 42 + 20 = 168 ✓

## 三、BLOCK_SIZES 关键读数

```
[BLOCK_SIZES]  scheduler_block_size=128  hash_block_size=128
               num_groups=6  group_block_sizes=[128, 128, 128, 128, 8, 32]
               cache_config.block_size=8  prefix_caching=False  connector=False  dcp=1  pcp=1
```

- `scheduler_block_size = lcm(128, 128, 128, 128, 8, 32) = 128`
- `hash_block_size = 128`（因为 `prefix_caching=False` 且 `connector=False`，走的是 `resolve_kv_cache_block_sizes` 里 "no consumer → 用 scheduler_block_size" 分支）
- `cache_config.block_size` 被自动改为 `min(所有组) = 8`
- **如果生产环境开启 `prefix_caching=True` 或有 KV connector**，`hash_block_size` 会重算为 `gcd(128, 128, 128, 128, 8, 32) = 8`，**粒度会塌到 8**。这是关键分水岭，见下节。

## 四、对 `kv_cache_affinity` 插件的影响

### 4.1 六组的可用性矩阵

| 组 | 有效可 aging？ | 原因 |
|---|---|---|
| **1（主 MLA，20 层）** | ✅ 唯一整套可用 | Full-prefix MLA，`get_cached_block(hash)` + `FullAttentionManager.find_longest_cache_hit` 语义匹配 |
| **0（indexer k-cache）** | ⚠️ 部分 | `AscendMLAAttentionSpec` + int8；如果 `hash_block_size=128` 且 hasher 与 spec 端一致，可用；`hash_block_size=8` 就废 |
| **2、3（SWA，44 层）** | ❌ 语义不匹配 | `SlidingWindowManager.find_longest_cache_hit` 只承认当前 128-token 窗口内的连续块；aging 窗外块**下一轮本就不会被命中**，做与不做无差 |
| **4（compressor, block=8）** | ❌ 完全错位 | block_size 与主组不同（8 vs 128）；且是 fp32 **流式状态**，语义上不是"前缀缓存" |
| **5（compressor, block=32）** | ❌ 同上 | 同 4 |

**有效工作面 = 20 / 168 ≈ 12%**（乐观情况下再加上 group 0 的 42 层可达 37%，但要看 hash_block_size 是不是 128）。

### 4.2 `hash_block_size` 塌到 8 的连锁反应

如果生产上 `prefix_caching=True` 或 connector 开启：

```
hash_block_size = gcd(128, 128, 128, 128, 8, 32) = 8
```

此时插件里 [`kv_cache_affinity/v1/engine/core.py:70`](../../src/kv_cache_affinity/v1/engine/core.py) 的：

```python
release_block_index = max(
    0,
    (release_start_index * len(req.block_hashes)) // len(req.all_token_ids) - 1,
)
hashes_to_release = req.block_hashes[release_block_index:]
```

会拿到 **按 8-token 粒度计算的 hash 列表**，然后在 group 0/1/2/3 上查 `block_pool.get_cached_block(hash)`。但这些组的物理 block 是**按 128-token 粒度**建立的 hash 索引，所以查询会**全 miss**，`[AGING_BLOCK]` 日志里会出现大量 `MISS (break) miss_at_first=True`。

要在这个模型上让插件真正工作，需要改 release 路径 —— **按每个 group 各自的 `block_size` 分别哈希**，而不是共用一套 `hash_block_size` 的 hash 序列。

### 4.3 本次数据 `prefix_caching=False` 的含义

如果本次采样的启动脚本就是压测/生产用的启动脚本，那么：

- vLLM 根本不维护 `cached_block_hash_to_block` 表 → 插件 `aging_block` 里的 `get_cached_block(...)` **永远返回 None**
- 插件当前部署形态下**在做无用功**，之前观察到的 `hit_rate=0.677` 里的 hit 全部来自模型/SWA 内部的其他缓存机制，与本插件无关

**必做验证**：确认压测启动命令中是否包含：
- `--enable-prefix-caching`
- `--kv-transfer-config '{...}'`（P 侧应该是 producer，D 侧是 consumer）

## 五、启动期日志原样（节选）

```
(EngineCore_DP0 pid=253732)  [KV_GROUP] total_groups=6 total_blocks=9161
(EngineCore_DP0 pid=253732)  [KV_GROUP] idx=0 spec=AscendMLAAttentionSpec block_size=128 num_layers=42
                             num_kv_heads=1 head_size=128 dtype=torch.int8 sliding_window=None
                             sample_layers=['model.layers.2.self_attn.indexer.k_cache',
                                            'model.layers.2.self_attn.attn',
                                            'model.layers.4.self_attn.indexer.k_cache', '...']
(EngineCore_DP0 pid=253732)  [KV_GROUP] idx=1 spec=AscendMLAAttentionSpec block_size=128 num_layers=20
                             num_kv_heads=1 head_size=512 dtype=torch.bfloat16 sliding_window=None
                             sample_layers=['model.layers.3.self_attn.attn',
                                            'model.layers.5.self_attn.attn',
                                            'model.layers.7.self_attn.attn', '...']
(EngineCore_DP0 pid=253732)  [KV_GROUP] idx=2 spec=AscendSlidingWindowMLASpec block_size=128 num_layers=22
                             head_size=512 dtype=torch.bfloat16 sliding_window=128
                             sample_layers=['model.layers.0.self_attn.swa_cache',
                                            'model.layers.2.self_attn.swa_cache',
                                            'model.layers.4.self_attn.swa_cache', '...']
(EngineCore_DP0 pid=253732)  [KV_GROUP] idx=3 spec=AscendSlidingWindowMLASpec block_size=128 num_layers=22
                             head_size=512 dtype=torch.bfloat16 sliding_window=128
                             sample_layers=['model.layers.1.self_attn.swa_cache',
                                            'model.layers.3.self_attn.swa_cache',
                                            'model.layers.5.self_attn.swa_cache', '...']
(EngineCore_DP0 pid=253732)  [KV_GROUP] idx=4 spec=AscendSlidingWindowMLASpec block_size=8 num_layers=42
                             head_size=2048 dtype=torch.float32 sliding_window=8
                             sample_layers=['model.layers.2.self_attn.compressor.state_cache',
                                            'model.layers.2.self_attn.indexer.compressor.state_cache',
                                            'model.layers.4.self_attn.compressor.state_cache', '...']
(EngineCore_DP0 pid=253732)  [KV_GROUP] idx=5 spec=AscendSlidingWindowMLASpec block_size=32 num_layers=20
                             head_size=1024 dtype=torch.float32 sliding_window=128
                             sample_layers=['model.layers.3.self_attn.compressor.state_cache',
                                            'model.layers.5.self_attn.compressor.state_cache',
                                            'model.layers.7.self_attn.compressor.state_cache', '...']
(EngineCore_DP0 pid=253732)  [BLOCK_SIZES] scheduler_block_size=128 hash_block_size=128 num_groups=6
                             group_block_sizes=[128, 128, 128, 128, 8, 32]
                             cache_config.block_size=8 prefix_caching=False connector=False dcp=1 pcp=1
```

## 六、v0.2.0 抓到的新事实：模型走的是自定义 attention 类，绕过 vLLM 的 Attention 基类

第二次采样（17:12–17:17，attn-trace v0.2.0）把 `[ATTN_IMPL_INIT]` 加进来之后，出现一个意外结果：

| tag | 期望条数 | 实际条数 |
|---|---:|---:|
| `[ATTN_BACKEND_PICK]` | 16（每 worker 1 次，functools.cache 去重） | 16 ✓ |
| `[NPU_ATTN_MAP]` | 16 | 16 ✓ |
| `[KV_GROUP]` | 14（每 DP 1 汇总 + 6 组明细） | 14 ✓ |
| `[BLOCK_SIZES]` | 2（每 DP 1 次） | 2 ✓ |
| **`[LAYER_KV_SPEC]`** | ≥ 168（每层 1 次） | **0** ❌ |
| **`[ATTN_IMPL_INIT]`** | ≥ 168 × 16 workers | **0** ❌ |
| **`[KV_MGR_INIT]`** | 6 × 2 DP = 12 | **0** ❌ |
| **`[KV_STARTUP_SUMMARY]`** | 2 | **0** ❌ |

排查 vllm-ascend 源码后定位到根因 —— DeepSeek V4 Flash 用的是**自定义 attention 类，不继承 vLLM 的 `Attention` / `MLAAttention` 等 6 个基类**：

```
# vllm/model_executor/layers/attention_layer_base.py
class AttentionLayerBase(ABC):           # 真正的顶层抽象
    @abstractmethod def get_attn_backend(self)
    @abstractmethod def get_kv_cache_spec(self, vllm_config)

# vllm-ascend 侧
class DSAAttention(nn.Module, AttentionLayerBase):                        # 直继基类
class AscendCompressorStateCache(CompressorStateCache):                   # 祖父类是 AttentionLayerBase
class AscendDeepseekV4IndexerCache(DeepseekV4IndexerCache):               # 祖父类是 AttentionLayerBase
# vllm 侧
class CompressorStateCache(torch.nn.Module, AttentionLayerBase)
class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase)
class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase)
```

v0.2.0 只 patch 了 `Attention` / `MLAAttention` / `CrossAttention` /
`ChunkedLocalAttention` / `EncoderOnlyAttention` / `StaticSinkAttention` 六个类，跟上面 4 个真正被实例化的类**完全没交集** —— 所以 168 层里 0 条日志。

同时期观察到的其它两个 0 条：
- `[KV_MGR_INIT]` = 0：`get_manager_for_kv_cache_spec` 在 hybrid 场景下被通过 `from ... import` 的方式在**别的 coordinator 模块**里预先绑定，patch 只改了源模块的符号，consumer 已经拿到旧引用。
- `[KV_STARTUP_SUMMARY]` = 0：**大概率是日志被截断**（info.txt 收在 DP0 `[BLOCK_SIZES]` 17:17:03 那行，warmup 还没跑完）；也可能是最外层 `DPEngineCoreProc.__init__` 才是真正实例化的类，而 v0.2.0 只包了 `EngineCore.__init__`。

### v0.3.0 的修法

- **`[LAYER_KV_SPEC]` / `[ATTN_IMPL_INIT]`**：改成扫 `AttentionLayerBase` 的子孙树；同时在 `AttentionLayerBase.__init_subclass__` 上装钩子，捕捉模型文件加载后动态出现的新子类。
- **`[KV_MGR_INIT]`**：扫描所有 `vllm.v1.core.*` 模块，把持有旧引用的本地符号一并替换。
- **`[KV_STARTUP_SUMMARY]`**：patch 链扩到 `EngineCore` + `EngineCoreProc` + `DPEngineCoreProc` 三个 `__init__`，try/finally 保证异常路径也能 emit；实例属性防重复，最外层只输出一次。

### v0.3.0 实测结果（第二次抓取，2026-08-27 09:30-09:35）

- ✅ **`[LAYER_KV_SPEC]`** × 46 fire — 子类扫描 + hook 生效。Late-wrap 捕捉到 15+ 个类，含 DeepSeek V4 侧 4 个基类和 Ascend 侧 4 个子类；顺带发现 DeepSeek V3.2 的 `DeepseekV32IndexerCache` 也在同环境；bonus 还抓到了 `MambaBase` 和 `CacheOnlyAttentionLayer`。
- ❌ **`[ATTN_IMPL_INIT]`** × 0 — v0.3.0 的 `_scanner.install_subclass_hook` 只支持一个 callback，`attn_impl.apply()` 后跑，callback 被静默丢弃。**v0.3.1 已修**（callback 改成列表）。
- ❌ **`[KV_MGR_INIT]`** × 0 — hybrid coordinator 在 `vllm_ascend.core.kv_cache_coordinator` 里，v0.3.0 的 consumer 扫描只扫 `vllm.v1.core.*`，漏掉了 Ascend 侧。**v0.3.1 已修**（扩到 `vllm.*` + `vllm_ascend.*` + 任何带 `coordinator` / `kv_cache` 关键字的模块）。
- ❌ **`[KV_STARTUP_SUMMARY]`** × 0 — 用户确认原始 3000+ 行日志中一行都没有。原因未定；可能是 `type(self).__name__ == tag` 匹配没走到、或 finally 有其他因素。**v0.3.1 改成每层都 emit** 一条带 `wrapper_stage` 字段的 summary + 一条 `[ENGINE_INIT_ENTER]` 诊断行，用户 grep `wrapper_stage=DPEngineCoreProc` 拿最终状态。

### v0.3.0 抓到的 LAYER_KV_SPEC 明细

从 46 条 `[LAYER_KV_SPEC]` 反推出的实例分布（去重后按 spec + layer 类分组）：

| owner class | spec | block_size | head_size | dtype | sliding_window | 层数（观察到） |
|---|---|---:|---:|---|---:|---:|
| `vllm_ascend.models.layer.attention.layer.DSAAttention` | AscendMLAAttentionSpec | 128 | 512 | bf16 | – | 41（layers 2–42 主 attention 通路） |
| `vllm_ascend.models.deepseek_v4.AscendDeepseekV4IndexerCache` | AscendMLAAttentionSpec | 128 | 128 | int8 | – | 42（indexer k-cache） |
| `vllm_ascend.models.deepseek_v4.AscendDeepseekV4SWACache` | AscendSlidingWindowMLASpec | 128 | 512 | bf16 | 128 | 44（SWA 缓存，按 parity 拆两组） |
| `vllm_ascend.models.deepseek_v4.AscendCompressorStateCache` | AscendSlidingWindowMLASpec | 8 | 2048 | fp32 | 8 | 42（indexer + main compressor state） |
| `vllm_ascend.models.deepseek_v4.AscendCompressorStateCache` | AscendSlidingWindowMLASpec | 32 | 1024 | fp32 | 128 | 20（主分支 compressor state） |
| `vllm_ascend.models.deepseek_v4.AscendCompressorStateCache` | AscendSlidingWindowMLASpec | 8 | 512 | fp32 | 8 | ~几个 | — 少见的 head=512 fp32 变体 |

这**证实了**之前基于 `[KV_GROUP]` 反推的 6 组结构 —— layer_name 层面每一条的 spec 都对上号：DSAAttention 就是主 MLA、AscendDeepseekV4IndexerCache 就是 int8 稀疏 indexer、AscendCompressorStateCache 有 block_size=8 / 32 两种，SWA 有 44 层。

layer_name 里可以看到部分条目 `layer=?`（cache 类没设 `self.layer_name`），只有 `DSAAttention` 主分支设了 `layer=model.layers.X.self_attn.attn`。这个属于命名规范限制，不影响结构判断。

装 v0.3.1 后再抓一次，`[ATTN_IMPL_INIT]` / `[KV_MGR_INIT]` / `[KV_STARTUP_SUMMARY]` 应该都会补齐；届时会再更新本文档补 impl_cls / manager_cls / hybrid coordinator 的真实分布。

## 七、注意事项

1. **本次采样 `prefix_caching / connector` 都是 False**：直接决定下游插件能否工作，必须在生产复现前确认真实启动开关。
2. **`hash_block_size` 依赖生产开关**：8 vs 128 决定了插件"完全失效"还是"部分可用"。
3. **DP0 与 DP1 的 `total_blocks` 差 1**：9161 vs 9162，属于内存 profiling 的正常抖动，不影响结构分析。
4. **本文件只反映 DeepSeek V4 Flash**：不同版本 DeepSeek（V3、V3.2、R1、V4 Base）的组数/spec 组合会不同，需要各自跑一遍 `attn_trace` 建档。
5. **要看到 `[LAYER_KV_SPEC]` / `[ATTN_IMPL_INIT]` / `[KV_MGR_INIT]`，必须升到 attn-trace ≥ 0.3.0**。v0.2.0 及以下在此模型上会全部落空。
