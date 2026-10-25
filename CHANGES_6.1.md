# V6.1 修改说明

## 1. 最大派发 Batch 与驻留容量分开

`EngineConfig.batch` 只约束单次模型执行请求数；新增 `max_active_seqs=None`，默认由 Target KV 容量限制驻留请求。
`_prefill_tick` 只处理 Target，并按 Batch/token budget 拼接完整 Prompt，必要时分块首请求。
`step` 优先接纳可执行的 Prefill，接纳受限后执行 ready 请求。
因此离线 workload 不再每完成一条请求就必须单独 Prefill 下一条。

## 2. 只有投机批次才准备 Draft

移除 `Sequence.prefill_draft`。初始/恢复 Prefill 都不预留请求 Draft 页、不执行 Draft Runner。
`_select` 冻结 B/K/每请求候选额度并原子预留两模型空间；回退 Target-only 的请求不触发 Draft。
`_sync_draft` 只由投机流程调用，检查 active batch 及候选额度；D=0 时先构建真实前缀，
已有缓存时保留 V5.1 的缺1/缺2 Graph 路径。其余缺口分块 eager 补算到 L-1，跳过 LM head。

## 3. 保留的执行路径

Graph 函数、eager attention 打包、KV 提交/回滚、greedy 接受规则、固定 KV 池大小沿用 V6.0。
模型初始化的 warmup/Graph capture 仍然执行；限制的是请求计算时机。
全局 Draft KV 池仍在初始化分配，页表直到真正投机才为请求预留。

## 4. 验证与来源

`tests/test_lazy_draft.py` 新增针对延迟执行与宽接纳的回归；原有测试仅修改活跃容量断言。
`tests/validate_lazy_cuda.py` 提供 A100 双模型短程验证，覆盖纯 Target-only 与冷 Draft 尾部投机。
所有来源文件以 `V6.0_BASE_SHA256.json` 固定；修改清单及验证状态随最终打包记录。
V6.0 原目录和原压缩包未覆盖。未执行自动 GPU 释放。

2026-10-24 验证完成：38 项 CPU 测试通过；A100-SXM4-80GB 上纯 Target-only
与自适应延迟 Draft workload 通过，32 条可同时驻留但每次只派发最多 16 条。
自适应测试仅 8 条请求触发 Draft，准备 616 token；同形状 eager/Graph 输出一致。
此回归不构成吞吐测量，尚无 V6.1 Spec-Bench 加速结论。
