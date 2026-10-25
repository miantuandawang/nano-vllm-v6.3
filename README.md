# nano-vllm V6.3

基于 V6.2，将默认活跃请求上限 `max_active_seqs` 从 128 调整为 372。
Python API、示例入口和 `bench.py`（复用示例参数解析器）均使用这一默认值。

| 参数 | V6.3 |
|---|---|
| `max_active_seqs` | 默认 372；实际驻留数量同时受 Target KV 空间约束 |
| `batch` | 默认 128，允许 1–128；每次调度最多 128 条 |
| Target Decode CUDA Graph | 1、2、4、8、16、32、48、64、80、96、112、128 |
| Target / Draft KV 配额 | 28 / 8 GiB |
| `target_only_threshold` | 16；实际调度 B≥16 时执行 Target-only |
| 上下文 / 单次 token 预算 / KV block | 4096 / 4096 / 256 |

372 限制的是已经接纳、正在 Prefill 或位于 ready 队列的请求总量。
Waiting 中未接纳的请求不占这个名额；单次实际执行仍最多调度 128 条。
ready 中的请求可以分多个批次执行，因此扩大活跃容量不要求扩大 Decode Graph 桶。
现有 Scheduler 已独立使用 `max_active_seqs` 和 `batch`，本版直接复用这套实现。

固定 KV 配额没有扩大，因此 `max_active_seqs=372` 是容量上限，不保证任意长度的
372 条请求都能同时驻留。KV 不足时沿用 V6.2 的等待、Target KV 抢占重算和
投机预留失败时退回 Target-only 的逻辑。

请求级 Draft Prefill/补算仍只在实际调度投机、完成双侧 KV 预留后执行。
Prefill 优先策略、接受/拒绝、KV 提交/回滚和 Graph 捕获/replay 算法沿用 V6.2。

## 使用

```bash
python -m pip install -e .
python example.py --target-model /models/Qwen3-14B \
  --draft-model /models/Qwen3-0.6B \
  --batch 128 --max-active-seqs 372 --prompt "Hello" --max-tokens 128
```

```python
from nanovllm import LLM, EngineConfig, SamplingParams

config = EngineConfig(
    target_model='/models/Qwen3-14B',
    draft_model='/models/Qwen3-0.6B',
    batch=128,
    max_active_seqs=372,
)
with LLM(config) as llm:
    outputs = llm.generate(['Hello'] * 480, SamplingParams(max_tokens=128))
```

## 验证与来源

本版执行了 Python 语法解析、API/CLI 默认参数检查、继承代码一致性和归档完整性检查。
未运行 CPU 状态机测试、CUDA 正确性测试或性能测试，未操作 Colab/GPU。

- `CHANGES_6.3.md`：本版修改说明。
- `V6.3.patch`：相对 V6.2 的 Python/项目配置差异。
- `V6.2_BASE_SHA256.json`：创建时的 V6.2 文件校验值。
- `VALIDATION_STATIC.json`：本版静态检查记录。
- `SOURCE_MANIFEST.json`：发布文件校验值（不含自身）。
- `V6.2_README.md`、`V6.2_TESTING_NOT_RUN.json`：V6.2 历史记录。
