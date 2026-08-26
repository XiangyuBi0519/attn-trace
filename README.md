# attn-trace

启动时把 vLLM 选中的 **attention backend / KV cache spec / KV manager / hash block 粒度**全打成一行行结构化日志，帮你判断当前模型 + 后端组合是否与你的插件（如 `kv_cache_affinity`）的 block-hash 语义匹配。

零侵入：不改 vLLM / vllm-ascend 源码，pip 装完由 vLLM 的 `vllm.general_plugins` 入口自动加载。

## 装

在你拉起服务的容器里：

```bash
pip install ./attn_trace          # 或 pip install /path/to/attn_trace-0.1.0-*.whl
```

## 打日志的位置（一次拉起触发）

| tag | 触发点 | 含义 |
|---|---|---|
| `[ATTN_BACKEND_PICK]` | `vllm/v1/attention/selector.py::_cached_get_attn_backend` | 每种 (head_size, dtype, use_mla, use_sparse, ...) 组合被选到的 backend 类 |
| `[NPU_ATTN_MAP]` | `vllm_ascend/platform.py::NPUPlatform.get_attn_backend_cls` | Ascend 侧分派到 MLA / SFA / DSA / FA3 / 普通 / 310p |
| `[LAYER_KV_SPEC]` | `Attention.get_kv_cache_spec`（及所有子类） | 每个 layer 生成的 spec，含 sliding_window / attention_chunk_size |
| `[ATTN_IMPL_INIT]` | `Attention.__init__`（及所有子类）末尾 | 每个 layer **实例化后**真正持有的 backend 名 + impl 类，跨模型对比 impl 差异用 |
| `[KV_GROUP]` | `EngineCore._initialize_kv_caches` 末尾 | 最终归并出的每个 KV cache group（spec 类、block_size、层数、样例 layer 名） |
| `[BLOCK_SIZES]` | `resolve_kv_cache_block_sizes` | scheduler_block_size / hash_block_size / 每组 block_size —— **决定你插件按 hash 重算 block 时对不对得上** |
| `[KV_MGR_INIT]` | `get_manager_for_kv_cache_spec` | 每个组实际用哪个 Manager 类（FullAttention / SlidingWindow / ChunkedLocal / Mamba / ...） |
| `[KV_STARTUP_SUMMARY]` | `EngineCore.__init__` 末尾 | 整机汇总：几组、每组几层、hash_block_size、prefix caching / kv connector 开关 |

## 环境变量

- `ATTN_TRACE_DISABLE=1`：完全关掉（不打日志、不 patch）
- `ATTN_TRACE_LOG_LEVEL=DEBUG|INFO|WARNING`：调整插件自己的 logger 级别，默认 `INFO`
- `ATTN_TRACE_LOG_FILE=/path/to/file.log`：把插件日志额外写到文件（追加），默认只往 stderr

## 手动触发（非 entry-point 场景）

如果你不走 vLLM 插件入口，也可以在自己的启动脚本里显式调：

```python
from attn_trace.plugin import register
register()
```

允许多次调用，第二次起是 no-op。

## 参考模型档案

抓到过的模型 KV 布局观测记录放在 [`docs/models/`](docs/models/) 里，方便跨模型对比。新增模型跑完后欢迎把 `[KV_STARTUP_SUMMARY]` / `[KV_GROUP]` / `[BLOCK_SIZES]` 三段整理成同格式 md 补进去。

- [DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) —— MLA + Sparse Attention（indexer + compressor）+ SWA，6 组 hybrid KV

## 与 `kv_cache_affinity` 的关系

`attn-trace` 只读不写：不改变任何 vLLM 行为，只把 backend / spec / manager / 粒度打日志。可以和 `kv_cache_affinity` 一起装、都通过 `vllm.general_plugins` 加载，各自独立。
