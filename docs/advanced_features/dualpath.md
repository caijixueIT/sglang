# DualPath in SGLang

## 1. Overview

This document describes the experimental DualPath implementation built on top of SGLang PD Disaggregation and HiCache. It serves three purposes:

1. Explain how the DualPath paper maps to the SGLang architecture.
2. Document the current implementation details, touched modules, and runtime behavior.
3. Provide reproducible test and benchmark steps, plus a concrete roadmap for the next phase.

The implementation target is the paper:

- `DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference`

In short, DualPath extends the classic `storage -> prefill` optimization path with a new `storage -> decode` path, and complements it with a reverse `decode -> prefill` KV transfer path so that prefill can reuse prompt KV that was already materialized on the decode side.

## 2. Motivation

In a conventional SGLang PD deployment:

- Prefill computes prompt KV.
- Prefill sends KV to decode.
- Decode performs generation.
- HiCache may write prompt/decode KV back to host or storage.

This architecture still leaves an important bottleneck:

- If useful KV has already been written to storage, the system mostly assumes it will be consumed again by prefill.
- Decode cannot fully exploit storage-side KV locality.
- Prefill must often recompute prompt KV even when decode-side offload/storage already contains reusable data.

DualPath addresses this by adding two new capabilities:

1. `storage -> decode`
   Decode can read KV from L3 storage into its host-side buffers and use those KV directly.

2. `decode -> prefill`
   Decode can reverse-transfer reusable prompt KV back to prefill, allowing prefill to shrink the amount of fresh prompt computation.

## 3. Scope of the Current Implementation

The current implementation reaches the level of an end-to-end experimental prototype with unit tests and a working pressure smoke test.

### Already implemented

- DualPath request metadata propagation across Python entrypoints and scheduler layers
- Decode-side storage read MVP using the existing HiCache storage/prefetch capabilities
- Bidirectional disaggregation transport abstraction for `decode -> prefill`
- Prefill-side reverse KV merge to shrink prompt compute
- Router-side DualPath metadata injection
- Decode-side independent bootstrap port for reverse transfer
- Observability and metrics for decode storage reads
- Unit tests for core control-plane and merge logic
- End-to-end pressure smoke validation with metric-based success criteria

### Not yet fully productized

- Full production-grade global scheduler like the paper
- Robust Rust gateway readiness synchronization for cold start
- Multi-node benchmark suite and larger-scale evaluation
- Fine-grained QoS isolation and full traffic-class validation under real RDMA

## 4. Design Mapping: Paper to SGLang

### 4.1 Components in the paper

The paper introduces three ideas:

- A classic `storage -> prefill` path
- A new `storage -> decode` path
- A reverse `decode -> prefill` transfer path

### 4.2 Matching SGLang concepts

These ideas map to the SGLang stack as follows:

- PD Disaggregation
  Provides the natural split between prefill and decode workers.

- HiCache
  Provides local GPU/host cache plus storage-backed L3 cache.

- Decode offload manager
  Provides a natural place to track decode-side host/storage KV state.

- Disaggregation transport layer
  Provides the data plane for KV transfer between prefill and decode.

- Model gateway / router
  Provides the control plane for DualPath metadata injection and path selection.

## 5. Architecture

### 5.1 Traditional PD path

Without DualPath:

1. Request enters router.
2. Router assigns a prefill worker and a decode worker.
3. Prefill computes prompt KV.
4. Prefill transfers KV to decode.
5. Decode generates output tokens.

### 5.2 DualPath-enabled path

With DualPath:

1. Router selects the path mode and injects DualPath metadata.
2. Decode may read reusable KV from storage into host buffers.
3. Decode may reverse-transfer reusable prompt KV to prefill.
4. Prefill tries to merge these prompt KV pages before extending the prompt.
5. Only the prompt miss portion is recomputed.
6. Decode continues normal generation using PD and HiCache mechanisms.

### 5.3 Current path modes

The implementation supports the following control-plane modes:

- `prefill_only`
  Force the classic path behavior.

- `decode_only`
  Favor the decode-side storage-read path.

- `hybrid_auto`
  Use the current heuristic auto-selection path.

## 6. Control Plane Changes

### 6.1 New server/runtime arguments

The implementation adds these arguments on the server side:

- `--dualpath-enable`
- `--dualpath-decode-bootstrap-port`
- `--dualpath-static-mode`
- `--dualpath-layer-streaming-chunk-pages`
- `--dualpath-ib-traffic-class`

These arguments enable DualPath and configure reverse-transfer/bootstrap behavior.

### 6.2 New request metadata

The following request fields are propagated across OpenAI-compatible requests, internal request structs, and scheduler `Req` objects:

- `dualpath_mode`
- `dualpath_selected_path`
- `dualpath_decode_bootstrap_host`
- `dualpath_decode_bootstrap_port`

These fields allow the router and runtime to coordinate DualPath selection and reverse-transfer endpoints.

### 6.3 Router behavior

The DualPath-aware router injects:

- selected path mode
- selected data path
- decode bootstrap address for reverse communication

The Rust PD router supports this directly. The Python binding layer was extended to parse the new CLI arguments and pass DualPath state into the runtime.

For the pressure smoke path, `MiniLB` was also extended to inject the same metadata so that testing does not depend on full Rust gateway readiness behavior.

## 7. Data Plane Changes

### 7.1 Bidirectional transfer abstraction

The disaggregation layer was extended so that sender/receiver roles are not hard-coded to prefill/decode identity.

Key addition:

- `KVTransferDirection`

This allows the system to support:

- `PREFILL_TO_DECODE`
- `DECODE_TO_PREFILL`

### 7.2 Decode-side storage read

Decode-side storage read is implemented as an MVP path on top of the existing decode offload / HiCache infrastructure:

1. Decode tracks offloaded KV in host/storage.
2. Decode initiates storage reads from HiCache L3.
3. Read results land in host-side buffers.
4. Decode may later reverse-transfer those buffers to prefill.

The main metric used for validation is:

- `decode_storage_read_hit_tokens`

### 7.3 Reverse `decode -> prefill` transfer

When DualPath is enabled:

- Decode creates a reverse sender using its host KV pool metadata.
- Prefill creates a reverse receiver with a decode bootstrap endpoint.
- Prefill sends destination page metadata.
- Decode sends reverse-transferred pages.
- Prefill merges the received pages into prompt processing.

### 7.4 Prompt KV merge on prefill

The key merge logic lives in `ScheduleBatch` and works as follows:

1. Prefill allocates normal output cache locations.
2. Prefill tries reverse-receive for the reusable prompt prefix.
3. On success:
   - `prefix_indices` are updated
   - `extend_input_len` is reduced
   - hit statistics are updated
4. Prefill only computes the miss suffix of the prompt

This makes the reverse path useful beyond mere transport correctness.

## 8. Observability and Metrics

To validate DualPath behavior, observability was extended in the scheduler/runtime metrics path.

Important disaggregation metrics include:

- `decode_offload_pending_reqs`
- `decode_backup_pending_reqs`
- `decode_storage_read_pending_reqs`
- `decode_storage_read_hit_tokens`

Worker-side load snapshots are available from:

- `/v1/loads?include=disagg`

The pressure smoke script validates DualPath by checking that:

- requests complete successfully
- `decode_storage_read_hit_tokens` increases during the test

## 9. Main Code Locations

The following files are the most important implementation entry points.

### 9.1 Request and metadata propagation

- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/entrypoints/openai/protocol.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/entrypoints/EngineBase.py`
- `python/sglang/srt/entrypoints/engine.py`
- `python/sglang/srt/managers/session_controller.py`

### 9.2 Scheduler and merge logic

- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/managers/scheduler.py`

### 9.3 Disaggregation runtime

- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/decode.py`
- `python/sglang/srt/managers/disagg_service.py`
- `python/sglang/srt/server_args.py`

### 9.4 Router / gateway / binding layer

- `sgl-model-gateway/src/routers/http/pd_router.rs`
- `sgl-model-gateway/src/main.rs`
- `sgl-model-gateway/bindings/python/src/sglang_router/router_args.py`
- `sgl-model-gateway/bindings/python/src/sglang_router/router.py`
- `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`

### 9.5 Tests and validation

- `test/registered/unit/managers/test_dualpath_prefill_merge.py`
- `test/registered/unit/managers/test_disaggregation_roles.py`
- `test/registered/unit/managers/test_io_struct.py`
- `test/registered/unit/server_args/test_server_args.py`
- `sgl-model-gateway/bindings/python/tests/test_arg_parser.py`
- `sgl-model-gateway/bindings/python/tests/test_router_config.py`
- `scripts/dualpath_pressure_smoke.py`

## 10. How to Reproduce

This section documents the exact validation flow used during implementation.

### 10.1 Environment assumptions

- Repository path: `/pfs-verdent/libaoguo/sglang`
- At least 2 GPUs
- Model available: `Qwen/Qwen2.5-1.5B-Instruct`
- Free local ports:
  - `31000`
  - `31100`
  - `31200`
  - `31500`
  - `31501`

### 10.2 Build and install router Python binding

```bash
cd /pfs-verdent/libaoguo/sglang/sgl-model-gateway/bindings/python
python3 -m pip install setuptools-rust maturin
maturin build --release --skip-auditwheel
python3 -m pip install --force-reinstall target/wheels/sglang_router-0.3.2-cp38-abi3-linux_x86_64.whl
```

### 10.3 Run SGLang unit tests

```bash
cd /pfs-verdent/libaoguo/sglang
PYTHONPATH="/pfs-verdent/libaoguo/sglang/python" \
python -m pytest -q \
  test/registered/unit/managers/test_dualpath_prefill_merge.py \
  test/registered/unit/managers/test_io_struct.py \
  test/registered/unit/managers/test_disaggregation_roles.py \
  test/registered/unit/server_args/test_server_args.py
```

Expected result from the validated run:

- `66 passed`
- `4 subtests passed`

### 10.4 Run router binding tests

```bash
cd /pfs-verdent/libaoguo/sglang/sgl-model-gateway/bindings/python
python3 -m pytest -q tests/test_arg_parser.py tests/test_router_config.py
```

Expected result from the validated run:

- `49 passed`
- `2 skipped`

### 10.5 Run the end-to-end DualPath pressure smoke

```bash
cd /pfs-verdent/libaoguo/sglang
PYTHONPATH="/pfs-verdent/libaoguo/sglang/python" \
python scripts/dualpath_pressure_smoke.py
```

What this script does:

- launches a prefill worker
- launches a decode worker
- launches a PD router through `sglang_router.launch_router`
- warms the system
- sends concurrent requests
- queries `/v1/loads?include=disagg`
- validates `decode_storage_read_hit_tokens`

### 10.6 Expected pressure smoke result

A successful run should print a JSON summary similar to:

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

Success criteria:

- `completed_requests == 16`
- `decode_storage_read_hit_tokens_delta > 0`

The second condition is the key DualPath proof point.

## 11. Manual Service Startup for Debugging

If you want to debug the stack manually instead of using the smoke script, use the following commands.

### 11.1 Prefill worker

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

### 11.2 Decode worker

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

### 11.3 Router (stable smoke path)

```bash
cd /pfs-verdent/libaoguo/sglang
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill http://127.0.0.1:31100 31500 \
  --decode http://127.0.0.1:31200 \
  --host 127.0.0.1 \
  --port 31000 \
  --dualpath-enable \
  --dualpath-mode decode_only
```

### 11.4 Send a test request

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

### 11.5 Query decode-side disaggregation metrics

```bash
curl "http://127.0.0.1:31200/v1/loads?include=disagg"
```

## 12. Known Limitations

### 12.1 Current auto routing is heuristic

The current `hybrid_auto` behavior is not yet the full scheduler proposed in the paper. It is a practical heuristic built on worker load snapshots and the newly added observability metrics.

### 12.2 Rust gateway cold-start behavior needs hardening

During testing, the real Rust gateway path could accept traffic before worker readiness had fully converged. This can produce temporary `503` responses. The pressure smoke therefore uses the `MiniLB` path for a more stable reproducible validation flow.

### 12.3 MiniLB currently derives decode bootstrap port implicitly

The smoke path derives the decode bootstrap port from the prefill bootstrap port plus one. This matches the current implementation assumptions, but custom decode bootstrap port scenarios should be validated separately.

### 12.4 Some warnings do not currently block success

During the validated runs, the following warnings appeared but did not block success:

- no RDMA devices found, fallback behavior used
- transfer engine overlapped memory region warning
- prefill bootstrap query warning on `127.0.0.1:8999`

These should still be cleaned up in a production-quality rollout.

## 13. Suggested Next Roadmap

This section is intended to support the next engineering phase.

### Phase 1: Hardening and cleanup

- Remove the remaining bootstrap warning path in the pressure smoke environment
- Support explicit decode bootstrap port end-to-end in every test path
- Improve cold-start router readiness synchronization
- Add stronger assertions for router path injection

### Phase 2: Scheduler and routing quality

- Replace the current heuristic `hybrid_auto` logic with a richer cost model
- Incorporate storage pressure, host-cache pressure, and queue depth in path selection
- Add path decision metrics and path decision logging

### Phase 3: Performance validation

- Add multi-node DualPath benchmarks
- Benchmark with real RDMA devices and traffic classes
- Compare:
  - prefill-only baseline
  - decode-only baseline
  - hybrid auto
- Track TTFT, TPOT, storage bandwidth, and cross-stage interference

### Phase 4: Production rollout readiness

- Define rollback and disable switches
- Define acceptance thresholds
- Add regression tests to CI or nightly benchmark jobs
- Document deployment guidance for production clusters

## 14. Acceptance Criteria for the Next Phase

The next milestone should be considered complete only if all of the following are true:

1. Rust gateway path is stable during cold start without depending on `MiniLB`.
2. End-to-end tests pass with explicit decode bootstrap port configuration.
3. DualPath routing decisions are observable and debuggable.
4. Multi-node benchmark data shows repeatable benefit over the baseline.
5. CI or scheduled validation includes at least one DualPath smoke test.

## 15. Summary

The current DualPath implementation in SGLang is no longer just a design sketch. It already provides:

- request metadata plumbing
- decode-side storage read
- reverse `decode -> prefill` KV transfer
- prompt KV merge on prefill
- observability and metric-based validation
- working unit tests
- a reproducible end-to-end pressure smoke test

This makes the current state suitable for continued hardening, routing refinement, and larger-scale benchmark evaluation.
