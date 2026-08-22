# attn-trace

只读探针，追踪 vLLM 0.23 里模型走了哪些 attention manager 类型 + 什么时候触发 `prepend=True` 路径。**不修改行为**，只加日志。

设计目的：定位「同一插件、不同模型行为不一致」这类问题时，先脱离业务插件把模型自身的 attention 路径打透，避免瞎猜。

## 安装

支持两种安装方式，都会注册成标准 vLLM plugin（`vllm.general_plugins` entry point），
**vLLM 启动时会在主进程和每个 EngineCore 子进程各调用一次 `attn_trace.plugin.register()`**，
多 DP / multiproc executor 场景自动全覆盖。

### 方式 1：pip 开发装（本地 dev / 快速迭代）

```bash
git clone https://github.com/XiangyuBi0519/attn-trace.git
cd attn-trace
pip install -e .
```

### 方式 2：复制到框架路径 + setup.py 安装（公司内部部署流程）

跟 `kv_cache_affinity` 同款安装方式：把 `attn_trace/` 目录 + `setup.py` + `requirements.txt`
一起放到目标环境的部署路径下，然后

```bash
pip install .
```

`pyproject.toml` 只声明了 build 后端（setuptools），所有元数据（name / version / entry_points /
依赖）都在 `setup.py` 里，两条路径都能干净地走通。

## 用法：无插件基线

**关键：先把 `kv_cache_affinity` 等业务插件从加载列表里去掉**，只留 `attn_trace`：

```bash
VLLM_PLUGINS=attn_trace vllm serve <model> ...
```

或者启动脚本里 `--enable-plugins attn_trace`（版本用词不同，看你现在怎么加载业务插件的，把它换掉/临时禁掉即可）。

## 看日志

启动后你会得到四类日志（都以 `attn_trace` 命名，容易 grep）：

### 1. `[MGR_INIT]` — 每个 manager 实例化时的元信息

```
[MGR_INIT] cls=SlidingWindowManager group_id=0 block_size=16 spec_type=SlidingWindowSpec spec_attrs={'sliding_window': 4096, 'num_kv_heads': 16, 'use_mla': False}
[MGR_INIT] cls=FullAttentionManager group_id=1 block_size=16 spec_type=FullAttentionSpec spec_attrs={'num_kv_heads': 16, 'use_mla': False}
```

### 2. `[COORD_INIT]` — KVCacheGroup 完整拓扑（启动阶段一次性）

```
[COORD_INIT] cls=HybridKVCacheCoordinator num_groups=2
[COORD_INIT]   group[0] manager=SlidingWindowManager spec=SlidingWindowSpec layers=['model.layers.0.self_attn.attn', ..., 'model.layers.61.self_attn.attn'] (total=32)
[COORD_INIT]   group[1] manager=FullAttentionManager spec=FullAttentionSpec layers=[...] (total=32)
```

**这一段就把模型架构完全暴露了**——用了哪些 attention 类型、每类各多少层、有没有 MLA、sliding window 多大。

### 3. `[SKIP_FIRST]` — 每种 manager 第一次进 skipped 分支的时机（运行时）

```
[SKIP_FIRST] SlidingWindowManager.get_num_skipped_tokens()=64 (total_computed=4160) — this manager WILL trigger prepend_n path
```

如果 5 分钟对话跑完都没看到这条，说明请求长度还没超过窗口，`remove_skipped_blocks → prepend_n` 走不到（但迟早会走到）。

### 4. `[PREPEND_FIRST]` — 每个调用点第一次触发 `free_blocks(prepend=True)`

```
[PREPEND_FIRST] free_blocks(prepend=True) called from .../single_type_kv_cache_manager.py:free:764 (n=3)
[PREPEND_FIRST] free_blocks(prepend=True) called from .../single_type_kv_cache_manager.py:remove_skipped_blocks:501 (n=7)
```

区分两个触发点：
- `free:764` = **`SlidingWindowManager.free()`**，每个请求结束都触发。
- `remove_skipped_blocks:501` = **窗口滑动触发**，长上下文才见。

后续同一调用点静默避免刷屏；需要频次统计的话把 `_first_free_prepend_seen` 那段去掉。

## 对照跑

按下面顺序跑一次，就能拿到确定性答案：

1. **模型 A + `VLLM_PLUGINS=attn_trace`（无业务插件）**：
   - 看 `[COORD_INIT]` 有没有 `SlidingWindowManager` / `ChunkedLocalAttentionManager` 之类。
   - 发几条长对话（超过它的窗口），看 `[SKIP_FIRST]` / `[PREPEND_FIRST]` 出不出。
2. **模型 B + `VLLM_PLUGINS=attn_trace`（无业务插件）**：
   - 同样两件事。
3. 对比两次日志。

三种可能的结论：

| 结果 | 解读 |
|---|---|
| A 只有 FullAttentionManager；B 有 SWA/Chunked | 是模型架构差异，问题彻底解释清楚 |
| 两个都有 SWA，但 A 测试请求没触发 SKIP_FIRST | 之前只是"运气好"没踩，其实一样有隐患 |
| 两个都没 SWA、都触发 PREPEND | 说明是别的路径（可能 vllm-ascend 自己加了些什么）导致，需要看 stack |

## 手动模式（不用 plugin）

如果不想装成 plugin，也可以在启动脚本里手动调用：

```python
import attn_trace
attn_trace.enable()   # 幂等；主进程调一次即可，但子进程需要各自调
```

注意手动模式下**子进程不会自动加载**，多 DP / multiproc executor 场景请务必走 plugin 路径。

## 卸载

```bash
pip uninstall attn-trace
```

调完就删，不留生产污染。

## License

Apache-2.0
