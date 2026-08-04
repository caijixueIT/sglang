"""CPU unit tests for DSPARK speculative decoding over the Kimi-K3
PP16-prefill PD shape (Step 1: decode without DCP).

Covers the transfer-layer enablers that let a PP-sharded prefill peer pair
with a speculative-decoding decode peer:

- draft-aware KV layer-id namespace (target global ids + num_layers+j draft
  ids) and its backward compatibility with positional pairing,
- the mooncake registration wire's per-entry decode item lens,
- byte-exact mixed-geometry sends: MLA target entries keep the whole-row
  copy while GQA draft entries are head-sliced per decode rank
  (prefill TP1 -> decode TP16).

Real Kimi-K3 / Kimi-K3-DSpark topology throughout: 93 target layers
(24 MLA full-attention), 5 draft layers, 16 draft KV heads, head_dim 64.
"""

import struct
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.srt.disaggregation.utils import (
    build_draft_kv_layer_ids,
    build_pd_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

NUM_LAYERS = 93
FULL_ATTN_LAYERS = [l - 1 for l in list(range(4, 93, 4)) + [93]]
PREFILL_PP = 16
DECODE_TP = 16

# Kimi-K3-DSpark draft: 5 qwen3-style GQA layers, 16 KV heads, head_dim 64.
DRAFT_LAYERS = 5
DRAFT_KV_HEADS = 16
DRAFT_HEAD_DIM = 64
DRAFT_BYTES_PER_HEAD = DRAFT_HEAD_DIM * 2  # bf16


def stage_full_attn_layers(pp_rank: int):
    start, end = get_pp_indices(NUM_LAYERS, pp_rank, PREFILL_PP)
    return [l for l in FULL_ATTN_LAYERS if start <= l < end]


class FakeDraftPool(SimpleNamespace):
    pass


class FakeHybridPool:
    def __init__(self, layer_ids):
        self._layer_ids = list(layer_ids)

    def get_kv_layer_ids(self):
        return list(self._layer_ids)


class TestDraftKvLayerIdNamespace(CustomTestCase):
    def test_mha_draft_ids_repeat_per_kv_group(self):
        draft_pool = FakeDraftPool(layer_num=DRAFT_LAYERS)
        ids = build_draft_kv_layer_ids(draft_pool, 2 * DRAFT_LAYERS, NUM_LAYERS)
        per_layer = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]
        self.assertEqual(ids, per_layer * 2)
        # Disjoint from every target layer id.
        self.assertFalse(set(ids) & set(range(NUM_LAYERS)))

    def test_mla_style_draft_ids_single_group(self):
        draft_pool = FakeDraftPool(layer_num=DRAFT_LAYERS)
        ids = build_draft_kv_layer_ids(draft_pool, DRAFT_LAYERS, NUM_LAYERS)
        self.assertEqual(ids, [NUM_LAYERS + j for j in range(DRAFT_LAYERS)])

    def test_ragged_entry_count_is_rejected(self):
        draft_pool = FakeDraftPool(layer_num=DRAFT_LAYERS)
        with self.assertRaisesRegex(ValueError, "whole number of layer"):
            build_draft_kv_layer_ids(draft_pool, DRAFT_LAYERS + 1, NUM_LAYERS)

    def test_combined_ids_for_last_pp_stage_with_draft(self):
        stage_layers = stage_full_attn_layers(PREFILL_PP - 1)
        target_pool = FakeHybridPool(stage_layers)
        draft_pool = FakeDraftPool(layer_num=DRAFT_LAYERS)
        ids = build_pd_kv_layer_ids(
            target_pool,
            draft_pool,
            len(stage_layers),
            2 * DRAFT_LAYERS,
            NUM_LAYERS,
        )
        self.assertEqual(
            ids,
            stage_layers
            + [NUM_LAYERS + j for j in range(DRAFT_LAYERS)] * 2,
        )

    def test_pool_without_layer_ids_keeps_positional_metadata(self):
        draft_pool = FakeDraftPool(layer_num=DRAFT_LAYERS)
        ids = build_pd_kv_layer_ids(
            object(), draft_pool, 24, 2 * DRAFT_LAYERS, NUM_LAYERS
        )
        self.assertEqual(ids, [])

    def test_equal_topology_id_pairing_matches_positional(self):
        # Live tp16-dspark regression guard: when both peers carry the same
        # combined id list (PP1, equal TP), id pairing must reproduce the
        # legacy positional mapping exactly.
        combined = FULL_ATTN_LAYERS + (
            [NUM_LAYERS + j for j in range(DRAFT_LAYERS)] * 2
        )
        pairs = build_transfer_entry_pairs(
            combined,
            combined,
            len(combined),
            len(combined),
            allow_positional_fallback=True,
        )
        self.assertEqual(pairs, [(i, i) for i in range(len(combined))])

    def test_pp_stage_with_draft_pairs_into_decode_combined_list(self):
        stage_layers = stage_full_attn_layers(PREFILL_PP - 1)
        draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]
        src_ids = stage_layers + draft_ids * 2
        dst_ids = FULL_ATTN_LAYERS + draft_ids * 2
        pairs = build_transfer_entry_pairs(
            src_ids, dst_ids, len(src_ids), len(dst_ids)
        )
        expected = [
            (i, FULL_ATTN_LAYERS.index(l)) for i, l in enumerate(stage_layers)
        ]
        n_t, n_full = len(stage_layers), len(FULL_ATTN_LAYERS)
        # Draft K group then draft V group, at their decode positions.
        expected += [(n_t + j, n_full + j) for j in range(DRAFT_LAYERS)]
        expected += [
            (n_t + DRAFT_LAYERS + j, n_full + DRAFT_LAYERS + j)
            for j in range(DRAFT_LAYERS)
        ]
        self.assertEqual(pairs, expected)

    def test_non_draft_pp_stage_pairs_target_only(self):
        # Capture-only stages register no draft entries; their target entries
        # must still resolve against the decode's combined list.
        stage_layers = stage_full_attn_layers(6)
        dst_ids = FULL_ATTN_LAYERS + (
            [NUM_LAYERS + j for j in range(DRAFT_LAYERS)] * 2
        )
        pairs = build_transfer_entry_pairs(
            stage_layers, dst_ids, len(stage_layers), len(dst_ids)
        )
        self.assertEqual(
            pairs,
            [(i, FULL_ATTN_LAYERS.index(l)) for i, l in enumerate(stage_layers)],
        )


class TestMooncakeRegisterWirePerEntryLens(CustomTestCase):
    def _base_msg(self):
        return [
            b"None",
            b"10.0.0.8",
            b"31000",
            b"session-1",
            b"".join(struct.pack("Q", 1000 + i) for i in range(3)),
            b"",
            b"",
            b"9",
            str(DECODE_TP).encode("ascii"),
            b"36864",
            b"",
            b"",
            b"",
            b"",
            b"",  # staging base ptr
            b"",  # staging total size
            b"1",  # dcp size
            b"0",  # dcp rank
        ]

    def test_per_entry_item_lens_roundtrip(self):
        msg = self._base_msg()
        lens = [36864] * 24 + [8192] * (2 * DRAFT_LAYERS)
        msg.append(b"".join(struct.pack("I", l) for l in lens))
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_kv_item_lens_per_entry, lens)

    def test_legacy_peer_without_per_entry_lens(self):
        info = KVArgsRegisterInfo.from_zmq(self._base_msg())
        self.assertIsNone(info.dst_kv_item_lens_per_entry)


class RecordingEngine:
    def __init__(self):
        self.blocks = []

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        self.blocks.extend(zip(src_addrs, dst_addrs, lengths))
        return 0


PAGE_SIZE = 64
MLA_TOKEN_LEN = 576  # fp8 MLA KV: kv_lora_rank 512 + qk_rope 64
DRAFT_SRC_TOKEN_LEN = DRAFT_KV_HEADS * DRAFT_BYTES_PER_HEAD  # TP1: all heads
DRAFT_DST_TOKEN_LEN = DRAFT_SRC_TOKEN_LEN // DECODE_TP  # TP16: one head


def make_last_stage_manager():
    """Prefill manager for the last PP stage: target MLA entries + draft
    K/V entries, exactly what PrefillBootstrapQueue registers there."""
    stage_layers = stage_full_attn_layers(PREFILL_PP - 1)
    n_target = len(stage_layers)
    draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]

    mgr = object.__new__(MooncakeKVManager)
    mgr.kv_args = SimpleNamespace(
        engine_rank=0,
        page_size=PAGE_SIZE,
        kv_data_ptrs=(
            [10_000_000 * (i + 1) for i in range(n_target)]
            + [5_000_000_000 + 50_000_000 * j for j in range(2 * DRAFT_LAYERS)]
        ),
        kv_item_lens=(
            [MLA_TOKEN_LEN * PAGE_SIZE] * n_target
            + [DRAFT_SRC_TOKEN_LEN * PAGE_SIZE] * (2 * DRAFT_LAYERS)
        ),
        kv_layer_ids=stage_layers + draft_ids * 2,
        prefill_start_layer=get_pp_indices(NUM_LAYERS, PREFILL_PP - 1, PREFILL_PP)[
            0
        ],
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
    return mgr, stage_layers


class TestMixedGeometryDraftSend(CustomTestCase):
    def test_last_stage_send_is_byte_exact_for_every_decode_rank(self):
        mgr, stage_layers = make_last_stage_manager()
        n_full = len(FULL_ATTN_LAYERS)
        draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]

        dst_kv_ptrs = [1_000_000_000 * (j + 1) for j in range(n_full)] + [
            9_000_000_000_000 + 90_000_000 * j for j in range(2 * DRAFT_LAYERS)
        ]
        dst_layer_ids = FULL_ATTN_LAYERS + draft_ids * 2
        dst_item_lens = [MLA_TOKEN_LEN * PAGE_SIZE] * n_full + [
            DRAFT_DST_TOKEN_LEN * PAGE_SIZE
        ] * (2 * DRAFT_LAYERS)

        src_pages = np.array([5, 2, 9], dtype=np.int32)
        dst_pages = np.array([7, 1, 3], dtype=np.int32)

        for dst_rank in range(DECODE_TP):
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
            self.assertEqual(ret, 0)

            # Expand every recorded block into (src_byte, dst_byte) pairs at
            # byte granularity and compare against a brute-force model.
            got = set()
            for src_addr, dst_addr, length in mgr.engine.blocks:
                got.add((src_addr, dst_addr, length))

            expected = set()
            # Target MLA entries: whole page-rows, coalesced into one block
            # (both page lists are non-contiguous, so 3 blocks per layer).
            for i, lid in enumerate(stage_layers):
                src_base = mgr.kv_args.kv_data_ptrs[i]
                dst_base = dst_kv_ptrs[FULL_ATTN_LAYERS.index(lid)]
                item_len = MLA_TOKEN_LEN * PAGE_SIZE
                for sp, dp in zip(src_pages, dst_pages):
                    expected.add(
                        (
                            src_base + int(sp) * item_len,
                            dst_base + int(dp) * item_len,
                            item_len,
                        )
                    )
            # Draft entries: one head-slice per token, at this decode rank's
            # head offset within the TP1 source row.
            n_t = len(stage_layers)
            for g in range(2 * DRAFT_LAYERS):
                src_base = mgr.kv_args.kv_data_ptrs[n_t + g]
                dst_base = dst_kv_ptrs[n_full + g]
                src_off = dst_rank * DRAFT_DST_TOKEN_LEN
                for sp, dp in zip(src_pages, dst_pages):
                    for t in range(PAGE_SIZE):
                        expected.add(
                            (
                                src_base
                                + int(sp) * DRAFT_SRC_TOKEN_LEN * PAGE_SIZE
                                + t * DRAFT_SRC_TOKEN_LEN
                                + src_off,
                                dst_base
                                + int(dp) * DRAFT_DST_TOKEN_LEN * PAGE_SIZE
                                + t * DRAFT_DST_TOKEN_LEN,
                                DRAFT_DST_TOKEN_LEN,
                            )
                        )
            self.assertEqual(got, expected)

    def test_all_ranks_together_consume_every_draft_head(self):
        mgr, stage_layers = make_last_stage_manager()
        n_full = len(FULL_ATTN_LAYERS)
        draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]
        dst_kv_ptrs = [1_000_000_000 * (j + 1) for j in range(n_full)] + [
            9_000_000_000_000 + 90_000_000 * j for j in range(2 * DRAFT_LAYERS)
        ]
        dst_layer_ids = FULL_ATTN_LAYERS + draft_ids * 2
        dst_item_lens = [MLA_TOKEN_LEN * PAGE_SIZE] * n_full + [
            DRAFT_DST_TOKEN_LEN * PAGE_SIZE
        ] * (2 * DRAFT_LAYERS)
        src_pages = np.array([0], dtype=np.int32)
        dst_pages = np.array([0], dtype=np.int32)

        first_draft_entry = len(stage_layers)
        src_base = mgr.kv_args.kv_data_ptrs[first_draft_entry]
        token0_bytes = set()
        for dst_rank in range(DECODE_TP):
            mgr.engine = RecordingEngine()
            mgr.send_kvcache(
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
            for src_addr, _, length in mgr.engine.blocks:
                if length != DRAFT_DST_TOKEN_LEN:
                    continue
                offset = src_addr - src_base
                if 0 <= offset < DRAFT_SRC_TOKEN_LEN:  # token 0 of entry 0
                    rng = set(range(offset, offset + length))
                    self.assertFalse(rng & token0_bytes, "head slices overlap")
                    token0_bytes |= rng
        # The 16 decode ranks tile the full 16-head source row exactly.
        self.assertEqual(len(token0_bytes), DRAFT_SRC_TOKEN_LEN)

    def test_missing_per_entry_lens_falls_back_to_whole_row(self):
        # Legacy decode peers (no per-entry lens) keep today's behavior:
        # every paired entry is copied at the src item len.
        mgr, stage_layers = make_last_stage_manager()
        # Equal-geometry decode: same entry lens as src.
        draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]
        n_full = len(FULL_ATTN_LAYERS)
        dst_kv_ptrs = [1_000_000_000 * (j + 1) for j in range(n_full)] + [
            9_000_000_000_000 + 90_000_000 * j for j in range(2 * DRAFT_LAYERS)
        ]
        dst_layer_ids = FULL_ATTN_LAYERS + draft_ids * 2
        src_pages = np.array([4], dtype=np.int32)
        dst_pages = np.array([6], dtype=np.int32)
        mgr.engine = RecordingEngine()
        ret = mgr.send_kvcache(
            "session",
            src_pages,
            list(dst_kv_ptrs),
            dst_pages,
            executor=None,
            dst_layer_ids=list(dst_layer_ids),
            dst_kv_item_lens=None,
            dst_tp_rank=3,
            dst_attn_tp_size=DECODE_TP,
        )
        self.assertEqual(ret, 0)
        lengths = sorted(length for _, _, length in mgr.engine.blocks)
        expected = sorted(
            [MLA_TOKEN_LEN * PAGE_SIZE] * len(stage_layers)
            + [DRAFT_SRC_TOKEN_LEN * PAGE_SIZE] * (2 * DRAFT_LAYERS)
        )
        self.assertEqual(lengths, expected)

    def test_upward_geometry_is_rejected(self):
        # dst entry larger than src (aggregation direction) must fail loudly.
        mgr, stage_layers = make_last_stage_manager()
        draft_ids = [NUM_LAYERS + j for j in range(DRAFT_LAYERS)]
        n_full = len(FULL_ATTN_LAYERS)
        dst_kv_ptrs = [1] * (n_full + 2 * DRAFT_LAYERS)
        dst_layer_ids = FULL_ATTN_LAYERS + draft_ids * 2
        dst_item_lens = [MLA_TOKEN_LEN * PAGE_SIZE] * n_full + [
            2 * DRAFT_SRC_TOKEN_LEN * PAGE_SIZE
        ] * (2 * DRAFT_LAYERS)
        mgr.engine = RecordingEngine()
        ret = mgr.send_kvcache(
            "session",
            np.array([0], dtype=np.int32),
            dst_kv_ptrs,
            np.array([0], dtype=np.int32),
            executor=None,
            dst_layer_ids=dst_layer_ids,
            dst_kv_item_lens=dst_item_lens,
            dst_tp_rank=0,
            dst_attn_tp_size=DECODE_TP,
        )
        self.assertEqual(ret, -1)


if __name__ == "__main__":
    unittest.main()
