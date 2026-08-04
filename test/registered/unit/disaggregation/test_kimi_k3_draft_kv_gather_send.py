"""CPU unit tests for the gathered draft-KV sliced send (mooncake).

The per-token sliced path posts one RDMA descriptor per token per draft
entry (64-128B each); the gathered path packs each decode rank's head slice
into a contiguous scratch and posts one descriptor per contiguous dst page
run. These tests pin, on the real Kimi-K3-DSpark draft geometry
(5 GQA layers, 16 KV heads, head_dim 64, prefill TP1 -> decode TP16):

- byte-for-byte equivalence with the per-token sliced path for every decode
  rank (dst byte address -> value maps are identical),
- the descriptor-count economy (entries x dst-page-runs, not x tokens),
- the per-token fallback when tensor handles are absent or a source pointer
  is unknown.
"""

import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

NUM_LAYERS = 93
FULL_ATTN_LAYERS = [l - 1 for l in list(range(4, 93, 4)) + [93]]
PREFILL_PP = 16
DECODE_TP = 16

DRAFT_LAYERS = 5
DRAFT_KV_HEADS = 16
DRAFT_HEAD_DIM = 64
DRAFT_DTYPE = torch.int16  # bf16-width, bit-preserving stand-in

PAGE_SIZE = 8
POOL_PAGES = 32
POOL_TOKENS = POOL_PAGES * PAGE_SIZE

MLA_TOKEN_LEN = 576
DRAFT_SRC_TOKEN_LEN = DRAFT_KV_HEADS * DRAFT_HEAD_DIM * 2
DRAFT_DST_TOKEN_LEN = DRAFT_SRC_TOKEN_LEN // DECODE_TP


def stage_full_attn_layers(pp_rank: int):
    start, end = get_pp_indices(NUM_LAYERS, pp_rank, PREFILL_PP)
    return [l for l in FULL_ATTN_LAYERS if start <= l < end]


class RecordingEngine:
    def __init__(self):
        self.blocks = []

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        self.blocks.extend(zip(src_addrs, dst_addrs, lengths))
        return 0


class FakeScratch:
    """CPU stand-in for the engine-registered StagingBuffer."""

    def __init__(self, size_bytes: int):
        self.buffer = torch.zeros(size_bytes, dtype=torch.uint8)

    def fits(self, required: int) -> bool:
        return required <= self.buffer.numel()

    def get_ptr(self) -> int:
        return self.buffer.data_ptr()

    def get_size(self) -> int:
        return self.buffer.numel()


def make_draft_tensors():
    torch.manual_seed(1234)
    return [
        torch.randint(
            -32768, 32767, (POOL_TOKENS, DRAFT_KV_HEADS, DRAFT_HEAD_DIM),
            dtype=DRAFT_DTYPE,
        )
        for _ in range(2 * DRAFT_LAYERS)
    ]


def make_last_stage_manager(draft_tensors, with_gather: bool):
    stage_layers = stage_full_attn_layers(PREFILL_PP - 1)
    n_target = len(stage_layers)
    draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]

    mgr = object.__new__(MooncakeKVManager)
    mgr.kv_args = SimpleNamespace(
        engine_rank=0,
        page_size=PAGE_SIZE,
        gpu_id=0,
        kv_data_ptrs=(
            [10_000_000 * (i + 1) for i in range(n_target)]
            + [t.data_ptr() for t in draft_tensors]
        ),
        kv_item_lens=(
            [MLA_TOKEN_LEN * PAGE_SIZE] * n_target
            + [DRAFT_SRC_TOKEN_LEN * PAGE_SIZE] * (2 * DRAFT_LAYERS)
        ),
        kv_layer_ids=stage_layers + draft_ids * 2,
        prefill_start_layer=get_pp_indices(NUM_LAYERS, PREFILL_PP - 1, PREFILL_PP)[0],
    )
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = True
    mgr.attn_tp_size = 1
    mgr.pp_size = PREFILL_PP
    mgr.pp_rank = PREFILL_PP - 1
    mgr.dcp_size = 1
    mgr.dcp_rank = 0
    mgr.enable_custom_mem_pool = False
    mgr.engine = RecordingEngine()
    mgr._sliced_gather_lock = threading.Lock()
    if with_gather:
        mgr._sliced_gather_buffers = {t.data_ptr(): t for t in draft_tensors}
        mgr._sliced_gather_scratch = FakeScratch(
            2 * DRAFT_LAYERS * POOL_TOKENS * DRAFT_DST_TOKEN_LEN
        )
    else:
        mgr._sliced_gather_buffers = {}
        mgr._sliced_gather_scratch = None
    return mgr, stage_layers


def make_dst_side(stage_layers):
    n_full = len(FULL_ATTN_LAYERS)
    draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]
    dst_kv_ptrs = [1_000_000_000 * (j + 1) for j in range(n_full)] + [
        9_000_000_000_000 + 90_000_000 * j for j in range(2 * DRAFT_LAYERS)
    ]
    dst_layer_ids = FULL_ATTN_LAYERS + draft_ids * 2
    dst_item_lens = [MLA_TOKEN_LEN * PAGE_SIZE] * n_full + [
        DRAFT_DST_TOKEN_LEN * PAGE_SIZE
    ] * (2 * DRAFT_LAYERS)
    return dst_kv_ptrs, dst_layer_ids, dst_item_lens


def run_send(mgr, dst_side, src_pages, dst_pages, dst_rank):
    dst_kv_ptrs, dst_layer_ids, dst_item_lens = dst_side
    mgr.engine = RecordingEngine()
    ret = mgr.send_kvcache(
        "session",
        src_pages,
        list(dst_kv_ptrs),
        dst_pages,
        executor=None,
        dst_layer_ids=list(dst_layer_ids),
        dst_kv_item_lens=list(dst_item_lens),
        dst_tp_rank=dst_rank,
        dst_attn_tp_size=DECODE_TP,
    )
    return ret, mgr.engine.blocks


def draft_dst_ptr_set(dst_side):
    dst_kv_ptrs, _, _ = dst_side
    lo = min(dst_kv_ptrs[-2 * DRAFT_LAYERS :])
    return lo


def dst_byte_value_map(blocks, resolve_src_byte, draft_dst_lo):
    """Expand draft blocks to a {dst_byte_addr: value} map."""
    out = {}
    for src_addr, dst_addr, length in blocks:
        if dst_addr < draft_dst_lo:
            continue  # MLA target entry (fake pointers, no memory behind them)
        for k in range(length):
            out[dst_addr + k] = resolve_src_byte(src_addr + k)
    return out


def tensor_byte_resolver(tensors):
    views = [(t.data_ptr(), t.numel() * t.element_size(), t) for t in tensors]

    def resolve(addr):
        for base, size, t in views:
            if base <= addr < base + size:
                flat = t.contiguous().view(torch.uint8).reshape(-1)
                return int(flat[addr - base])
        raise AssertionError(f"src byte 0x{addr:x} not inside any draft tensor")

    return resolve


def scratch_byte_resolver(scratch):
    base = scratch.get_ptr()
    size = scratch.get_size()

    def resolve(addr):
        assert base <= addr < base + size, "src byte outside scratch"
        return int(scratch.buffer[addr - base])

    return resolve


class TestGatheredDraftSend(CustomTestCase):
    def test_byte_equivalence_with_per_token_path_every_rank(self):
        draft_tensors = make_draft_tensors()
        src_pages = np.array([5, 2, 9], dtype=np.int32)
        dst_pages = np.array([7, 1, 3], dtype=np.int32)

        for dst_rank in range(DECODE_TP):
            mgr_g, stage_layers = make_last_stage_manager(
                draft_tensors, with_gather=True
            )
            dst_side = make_dst_side(stage_layers)
            ret_g, blocks_g = run_send(mgr_g, dst_side, src_pages, dst_pages, dst_rank)
            self.assertEqual(ret_g, 0)

            mgr_t, _ = make_last_stage_manager(draft_tensors, with_gather=False)
            ret_t, blocks_t = run_send(mgr_t, dst_side, src_pages, dst_pages, dst_rank)
            self.assertEqual(ret_t, 0)

            draft_lo = draft_dst_ptr_set(dst_side)
            got = dst_byte_value_map(
                blocks_g, scratch_byte_resolver(mgr_g._sliced_gather_scratch), draft_lo
            )
            want = dst_byte_value_map(
                blocks_t, tensor_byte_resolver(draft_tensors), draft_lo
            )
            self.assertEqual(len(want), 2 * DRAFT_LAYERS * 3 * PAGE_SIZE * DRAFT_DST_TOKEN_LEN)
            self.assertEqual(got, want, f"byte mismatch at dst_rank={dst_rank}")

    def test_descriptor_economy(self):
        draft_tensors = make_draft_tensors()
        src_pages = np.array([5, 2, 9], dtype=np.int32)  # 3 non-contiguous runs
        dst_pages = np.array([7, 1, 3], dtype=np.int32)

        mgr, stage_layers = make_last_stage_manager(draft_tensors, with_gather=True)
        dst_side = make_dst_side(stage_layers)
        draft_lo = draft_dst_ptr_set(dst_side)

        ret, blocks = run_send(mgr, dst_side, src_pages, dst_pages, dst_rank=3)
        self.assertEqual(ret, 0)
        draft_blocks = [b for b in blocks if b[1] >= draft_lo]
        # One descriptor per (entry x dst page run): 10 x 3, NOT 10 x tokens.
        self.assertEqual(len(draft_blocks), 2 * DRAFT_LAYERS * 3)
        for _src, _dst, length in draft_blocks:
            self.assertEqual(length, DRAFT_DST_TOKEN_LEN * PAGE_SIZE)

        # Contiguous page runs coalesce into a single descriptor per entry.
        src_pages_c = np.array([4, 5, 6], dtype=np.int32)
        dst_pages_c = np.array([11, 12, 13], dtype=np.int32)
        ret, blocks = run_send(mgr, dst_side, src_pages_c, dst_pages_c, dst_rank=3)
        self.assertEqual(ret, 0)
        draft_blocks = [b for b in blocks if b[1] >= draft_lo]
        self.assertEqual(len(draft_blocks), 2 * DRAFT_LAYERS)
        for _src, _dst, length in draft_blocks:
            self.assertEqual(length, 3 * DRAFT_DST_TOKEN_LEN * PAGE_SIZE)

    def test_per_token_fallback_without_tensor_handles(self):
        draft_tensors = make_draft_tensors()
        src_pages = np.array([5, 2, 9], dtype=np.int32)
        dst_pages = np.array([7, 1, 3], dtype=np.int32)

        mgr, stage_layers = make_last_stage_manager(draft_tensors, with_gather=False)
        dst_side = make_dst_side(stage_layers)
        draft_lo = draft_dst_ptr_set(dst_side)

        ret, blocks = run_send(mgr, dst_side, src_pages, dst_pages, dst_rank=3)
        self.assertEqual(ret, 0)
        draft_blocks = [b for b in blocks if b[1] >= draft_lo]
        self.assertEqual(len(draft_blocks), 2 * DRAFT_LAYERS * 3 * PAGE_SIZE)

    def test_fallback_on_unknown_source_pointer(self):
        draft_tensors = make_draft_tensors()
        src_pages = np.array([5, 2], dtype=np.int32)
        dst_pages = np.array([7, 1], dtype=np.int32)

        mgr, stage_layers = make_last_stage_manager(draft_tensors, with_gather=True)
        # Drop one entry's handle: the whole sliced batch must fall back
        # (partial gathered sends would double-write some entries).
        del mgr._sliced_gather_buffers[draft_tensors[3].data_ptr()]
        dst_side = make_dst_side(stage_layers)
        draft_lo = draft_dst_ptr_set(dst_side)

        ret, blocks = run_send(mgr, dst_side, src_pages, dst_pages, dst_rank=0)
        self.assertEqual(ret, 0)
        draft_blocks = [b for b in blocks if b[1] >= draft_lo]
        self.assertEqual(len(draft_blocks), 2 * DRAFT_LAYERS * 2 * PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
