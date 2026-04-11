# SGLang 中的 DualPath

## 1. 概述

本文档描述了构建在 SGLang PD Disaggregation 与 HiCache 之上的实验性 DualPath 实现，主要用于三个目的：

1. 说明论文中的 DualPath 思路如何映射到 SGLang 架构。
2. 记录当前实现的设计细节、涉及模块与运行时行为。
3. 提供可复现的测试与压测步骤，并给出下一阶段的 roadmap。

本次实现对应的论文为：

- `DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference`

简而言之，DualPath 在传统的 `storage -> prefill` 路径之外，引入了新的 `storage -> decode` 路径，并补充了 `decode -> prefill` 的反向 KV 传输路径，使 prefill 可以复用 decode 侧已经物化出来的 prompt KV。

## 2. 背景与动机

在常规的 SGLang PD 部署中：

- Prefill 负责计算 prompt KV。
- Prefill 将 KV 传给 decode。
- Decode 负责生成输出 token。
- HiCache 可能会把 prompt/decode 产生的 KV 回写到 host 或 storage。

但这种架构仍存在明显瓶颈：

- 即使有价值的 KV 已经写入 storage，系统默认也主要把它当作 `storage -> prefill` 的复用来源。
- Decode 侧不能充分利用 storage 中已有的 KV 局部性。
- 即使 decode 侧 offload/storage 中已有可复用 KV，prefill 仍然经常需要重新计算 prompt。

DualPath 通过新增两条能力来解决这个问题：

1. `storage -> decode`
   允许 decode 直接从 L3 storage 读取 KV 到 host buffer，并直接消费这些 KV。

2. `decode -> prefill`
   允许 decode 将可复用的 prompt KV 反向传回 prefill，从而减少 prefill 需要重新计算的 prompt 部分。

## 3. 当前实现范围

当前实现已经达到“可端到端跑通的实验原型”阶段，并配有单元测试和可工作的 pressure smoke test。

### 已完成部分

- DualPath 请求元数据在 Python entrypoint 与调度层之间的贯通
- 基于现有 HiCache storage/prefetch 能力的 decode 侧 storage read MVP
- 支持 `decode -> prefill` 的双向 disaggregation transport 抽象
- prefill 侧 reverse KV merge，用于缩短 prompt 计算
- router 侧 DualPath 元数据注入
- decode 侧独立 bootstrap 端口，用于 reverse transfer
- decode storage read 的观测指标与 metrics
- 控制面与 merge 逻辑的单元测试
- 通过指标判定的端到端 pressure smoke 验证

### 尚未完全产品化的部分

- 论文中那种完整的生产级全局调度器
- Rust gateway 冷启动阶段的强健 readiness 同步
- 多机 benchmark 套件与更大规模评估
- 基于真实 RDMA 的细粒度 QoS 隔离与完整验证

## 4. 论文设计到 SGLang 的映射

### 4.1 论文里的三个核心点

论文引入了三条关键路径：

- 传统 `storage -> prefill`
- 新增 `storage -> decode`
- 反向 `decode -> prefill`

### 4.2 在 SGLang 中的对应关系

这些概念在 SGLang 中的映射如下：

- PD Disaggregation
  天然提供 prefill 与 decode 的角色拆分。

- HiCache
  提供 GPU/host 本地缓存与 storage-backed 的 L3 缓存。

- Decode offload manager
  是跟踪 decode 侧 host/storage KV 状态的自然落点。

- Disaggregation transport layer
  是 prefill 与 decode 之间进行 KV 传输的数据面。

- Model gateway / router
  是注入 DualPath 元数据和进行路径选择的控制面。

## 5. 整体架构

### 5.1 传统 PD 路径

在没有 DualPath 时：

1. 请求先进入 router。
2. Router 为请求分配 prefill worker 和 decode worker。
3. Prefill 计算 prompt KV。
4. Prefill 将 KV 传给 decode。
5. Decode 继续生成输出 token。

### 5.2 引入 DualPath 后的路径

开启 DualPath 后：

1. Router 选择 path mode，并给请求注入 DualPath 元数据。
2. Decode 可以先从 storage 读出可复用的 KV 到 host buffer。
3. Decode 可以把可复用的 prompt KV 反向传回 prefill。
4. Prefill 在 extend prompt 前，先尝试合并这部分 reverse-transferred KV。
5. 只有 prompt miss 的那一段才会被重新计算。
6. Decode 再继续通过常规 PD/HiCache 机制完成生成。

### 5.3 当前支持的路径模式

当前控制面支持以下模式：

- `prefill_only`
  强制走传统路径。

- `decode_only`
  优先使用 decode 侧 storage read 路径。

- `hybrid_auto`
  使用当前启发式自动选路逻辑。

## 6. 控制面改动

### 6.1 新增 server/runtime 参数

当前实现引入了以下参数：

- `--dualpath-enable`
- `--dualpath-decode-bootstrap-port`
- `--dualpath-static-mode`
- `--dualpath-layer-streaming-chunk-pages`
- `--dualpath-ib-traffic-class`

这些参数用于开启 DualPath，并配置 reverse transfer 与 bootstrap 行为。

### 6.2 新增请求元数据

以下字段会在 OpenAI 兼容请求、内部请求结构和调度层 `Req` 对象之间进行透传：

- `dualpath_mode`
- `dualpath_selected_path`
- `dualpath_decode_bootstrap_host`
- `dualpath_decode_bootstrap_port`

这些字段用于协调 router 选路结果以及 reverse transfer 端点信息。

### 6.3 Router 行为

DualPath 感知的 router 会注入以下信息：

- 当前 path mode
- 当前选择的数据路径
- reverse 通信所需的 decode bootstrap 地址

Rust 版 PD router 已支持这部分逻辑。Python binding 层也扩展了相应 CLI 参数解析，并将 DualPath 状态传给运行时。

在 pressure smoke 测试路径中，`MiniLB` 也被扩展为注入相同的元数据，这样测试不再依赖完整 Rust gateway 的 readiness 稳定性。

## 7. 数据面改动

### 7.1 双向传输抽象

disaggregation 层被扩展为不再把 sender/receiver 角色硬编码绑定到 prefill/decode 身份上。

关键新增为：

- `KVTransferDirection`

它让系统可以显式支持：

- `PREFILL_TO_DECODE`
- `DECODE_TO_PREFILL`

### 7.2 Decode 侧 storage read

decode 侧 storage read 作为 MVP 路径，构建在现有 decode offload / HiCache 基础设施之上：

1. Decode 跟踪已经 offload 到 host/storage 的 KV。
2. Decode 从 HiCache L3 发起 storage read。
3. 读取结果落到 host-side buffer。
4. Decode 之后可以把这些 buffer 再反向传给 prefill。

当前验证这条路径是否生效的核心指标是：

- `decode_storage_read_hit_tokens`

### 7.3 `decode -> prefill` 反向传输

当 DualPath 开启后：

- Decode 使用自己的 host KV pool 元数据构造 reverse sender。
- Prefill 使用 decode bootstrap endpoint 构造 reverse receiver。
- Prefill 先发送目标页元数据。
- Decode 再发送 reverse-transferred pages。
- Prefill 将收到的页合并进 prompt 处理流程。

### 7.4 Prefill 侧 prompt KV merge

核心 merge 逻辑位于 `ScheduleBatch` 中，执行流程如下：

1. Prefill 先正常分配 output cache 位置。
2. Prefill 对可复用的 prompt prefix 尝试执行 reverse receive。
3. 如果成功：
   - 更新 `prefix_indices`
   - 减少 `extend_input_len`
   - 更新 hit 统计
4. Prefill 只对未命中的 prompt suffix 执行真实计算

这使得 reverse path 不只是“传输可用”，而是真正带来 prompt 计算缩减。

## 8. 可观测性与指标

为了验证 DualPath 是否真正生效，调度层与运行时指标链路被扩展了相应观测项。

关键 disaggregation 指标包括：

- `decode_offload_pending_reqs`
- `decode_backup_pending_reqs`
- `decode_storage_read_pending_reqs`
- `decode_storage_read_hit_tokens`

worker 侧可通过以下接口获取带 disaggregation 细节的负载快照：

- `/v1/loads?include=disagg`

pressure smoke 脚本通过以下方式判断 DualPath 是否工作：

- 请求必须全部完成
- `decode_storage_read_hit_tokens` 在测试期间必须增长

## 9. 关键代码位置

以下文件是当前实现的主要入口。

### 9.1 请求与元数据透传

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/entrypoints/openai/protocol.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/entrypoints/EngineBase.py`
- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/managers/session_controller.py`

### 9.2 调度与 merge 逻辑

- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`

### 9.3 Disaggregation 运行时

- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/managers/disagg_service.py`
- `python/sglang/srt/server_args.py`

### 9.4 Router / gateway / binding 层

- `sgl-model-gateway/src/routers/http/pd_router.rs`
- `sgl-model-gateway/src/main.rs`
- `sgl-model-gateway/bindings/python/src/sglang_router/router_args.py`
- `sgl-model-gateway/bindings/python/src/sglang_router/router.py`
- `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`

### 9.5 测试与验证

- `test/registered/unit/managers/test_dualpath_prefill_merge.py`
- `test/registered/unit/managers/test_disaggregation_roles.py`
- `test/registered/unit/managers/test_io_struct.py`
- `test/registered/unit/server_args/test_server_args.py`
- `sgl-model-gateway/bindings/python/tests/test_arg_parser.py`
- `sgl-model-gateway/bindings/python/tests/test_router_config.py`
- `scripts/dualpath_pressure_smoke.py`

## 10. 如何复现

这一节记录了实现过程中实际使用过的验证流程。

### 10.1 环境假设

- 仓库路径：`/pfs-verdent/libaoguo/sglang`
- 至少 2 张 GPU
- 可用模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 本地空闲端口：
  - `31000`
  - `31100`
  - `31200`
  - `31500`
  - `31501`

### 10.2 构建并安装 router Python binding

```bash
cd /pfs-verdent/libaoguo/sglang/sgl-model-gateway/bindings/python
python3 -m pip install setuptools-rust maturin
maturin build --release --skip-auditwheel
python3 -m pip install --force-reinstall target/wheels/sglang_router-0.3.2-cp38-abi3-linux_x86_64.whl
```

### 10.3 运行 SGLang 单元测试

```bash
cd /pfs-verdent/libaoguo/sglang
PYTHONPATH="/pfs-verdent/libaoguo/sglang/python" \
python -m pytest -q \
  test/registered/unit/managers/test_dualpath_prefill_merge.py \
  test/registered/unit/managers/test_io_struct.py \
  test/registered/unit/managers/test_disaggregation_roles.py \
  test/registered/unit/server_args/test_server_args.py
```

已验证通过的结果应类似：

- `66 passed`
- `4 subtests passed`

### 10.4 运行 router binding 测试

```bash
cd /pfs-verdent/libaoguo/sglang/sgl-model-gateway/bindings/python
python3 -m pytest -q tests/test_arg_parser.py tests/test_router_config.py
```

已验证通过的结果应类似：

- `49 passed`
- `2 skipped`

### 10.5 运行端到端 DualPath pressure smoke

```bash
cd /pfs-verdent/libaoguo/sglang
PYTHONPATH="/pfs-verdent/libaoguo/sglang/python" \
python scripts/dualpath_pressure_smoke.py
```

该脚本会执行：

- 拉起一个 prefill worker
- 拉起一个 decode worker
- 通过 `sglang_router.launch_router` 拉起一个 PD router
- 预热系统
- 发送并发请求
- 查询 `/v1/loads?include=disagg`
- 检查 `decode_storage_read_hit_tokens`

### 10.6 预期 pressure smoke 结果

成功运行时应打印出类似以下 JSON：

```json
{
  "completed_requests": 16,
  "decode_storage_read_hit_tokens_after": 35328,
  "decode_storage_read_hit_tokens_before": 10752,
  "decode_storage_read_hit_tokens_delta": 24576,
  "elapsed_s": 1.324,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "qps": 12.083
}
```

成功标准：

- `completed_requests == 16`
- `decode_storage_read_hit_tokens_delta > 0`

第二条是当前证明 DualPath 生效的关键证据。

## 11. 手动拉起服务进行调试

如果你想手动调试，而不是直接跑 smoke 脚本，可以按下面方式拉起服务。

### 11.1 启动 prefill worker

```bash
cd /pfs-verdent/libaoguo/sglang
PYTHONPATH="/pfs-verdent/libaoguo/sglang/python" \
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 31100 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 31500 \
  --tp 1 \
  --page-size 16 \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-ratio 2 \
  --dualpath-enable
```

### 11.2 启动 decode worker

```bash
cd /pfs-verdent/libaoguo/sglang
PYTHONPATH="/pfs-verdent/libaoguo/sglang/python" \
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 31200 \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 31500 \
  --base-gpu-id 1 \
  --tp 1 \
  --page-size 16 \
  --hicache-storage-backend file \
  --hicache-ratio 2 \
  --disaggregation-decode-enable-offload-kvcache \
  --num-reserved-decode-tokens 128 \
  --dualpath-enable
```

### 11.3 启动 router（稳定 smoke 路径）

```bash
cd /pfs-verdent/libaoguo/sglang
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill http://127.0.0.1:31100 \
  --decode http://127.0.0.1:31200 \
  --host 127.0.0.1 \
  --port 31000 \
  --dualpath-enable \
  --dualpath-mode decode_only
```

### 11.4 发送测试请求

```bash
curl -X POST http://127.0.0.1:31000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "DualPath manual test prompt. DualPath manual test prompt. DualPath manual test prompt.",
    "max_tokens": 32,
    "temperature": 0
  }'
```

### 11.5 查询 decode 侧 disaggregation 指标

```bash
curl "http://127.0.0.1:31200/v1/loads?include=disagg"
```

## 12. 当前已知限制

### 12.1 自动选路目前仍是启发式实现

当前的 `hybrid_auto` 逻辑还不是论文中完整意义上的全局调度器，而是建立在 worker load snapshot 与新增观测指标之上的一套实用启发式策略。

### 12.2 Rust gateway 冷启动路径还需要加强

在测试中，完整 Rust gateway 路径可能会在 worker readiness 尚未完全稳定时提前接收流量，从而返回临时性的 `503`。因此当前 pressure smoke 使用 `MiniLB` 路径，以获得更稳定、可复现的验证结果。

### 12.3 MiniLB 目前隐式推导 decode bootstrap port

当前 smoke 路径默认通过“prefill bootstrap port + 1”的方式推导 decode bootstrap port。这与当前实现假设一致，但如果需要自定义 decode bootstrap port，仍建议单独验证。

### 12.4 某些 warning 目前不阻塞成功运行

在已验证通过的运行中，出现过以下 warning，但并未阻止测试成功：

- 未发现 RDMA 设备，走了 fallback 路径
- transfer engine overlapped memory region warning
- 指向 `127.0.0.1:8999` 的 prefill bootstrap query warning

这些问题在生产化之前仍然应该继续清理。

## 13. 下一步建议 Roadmap

这一节用于支撑下一阶段开发。

### 阶段 1：稳定性与清理

- 清掉 pressure smoke 环境中剩余的 bootstrap warning
- 在所有测试路径中完整支持显式 decode bootstrap port
- 提升 cold-start 时 router readiness 同步的稳定性
- 对 router path injection 增加更强断言

### 阶段 2：调度与选路质量

- 用更丰富的 cost model 替换当前启发式 `hybrid_auto`
- 将 storage 压力、host cache 压力、queue depth 等因素纳入选路
- 增加 path decision metrics 与 path decision logging

### 阶段 3：性能验证

- 增加多机 DualPath benchmark
- 在真实 RDMA 设备与 traffic class 条件下做 benchmark
- 对比以下几组模式：
  - prefill-only baseline
  - decode-only baseline
  - hybrid auto
- 跟踪 TTFT、TPOT、storage bandwidth 与跨阶段干扰

### 阶段 4：上线准备

- 定义 rollback / disable 开关
- 定义 acceptance thresholds
- 将回归测试接入 CI 或 nightly benchmark
- 补齐生产环境部署说明

## 14. 下一阶段验收标准

下一里程碑至少要满足以下条件，才可以认为完成：

1. Rust gateway 路径在冷启动期间无需依赖 `MiniLB` 也能稳定工作。
2. 端到端测试在显式 decode bootstrap port 配置下可以稳定通过。
3. DualPath 的路径选择决策可观测、可调试。
4. 多机 benchmark 数据能稳定显示相对 baseline 的收益。
5. CI 或定时验证中至少包含一个 DualPath smoke test。

## 15. 总结

当前的 SGLang DualPath 实现已经不是概念验证草图，而是一个具备实际运行能力的实验原型，已经具备：

- 请求元数据透传
- decode 侧 storage read
- `decode -> prefill` 反向 KV 传输
- prefill 侧 prompt KV merge
- 指标与观测能力
- 单元测试
- 可复现的端到端 pressure smoke test

这意味着当前状态已经适合继续向“稳定性加固、选路优化与更大规模性能评估”推进。
