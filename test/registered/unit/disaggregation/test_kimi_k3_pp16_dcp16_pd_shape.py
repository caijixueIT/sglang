"""CPU unit tests pinning the Kimi-K3 PD deployment shape:

    prefill  TP1 x EP1 x PP16   <->   decode  TP16 x EP16 x DCP16 x PP1

over the mooncake transfer backend. Every test uses the real Kimi-K3
topology (93 hidden layers, 24 MLA full-attention layers, 69 KDA linear
layers, 96 linear-attention heads) so a regression in any of the shape's
load-bearing code paths -- decode->prefill rank mapping, the DCP token
relayout, PP layer-id entry pairing, the KDA state TP1->TP16 slicing, and
the PP-share mamba budget charge -- fails here instead of on the cluster.

All tests are pure-CPU: mooncake engine calls are recorded by fakes and
verified byte-exactly against a brute-force reference.
"""

import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.common.utils import (
    build_dcp_token_transfer_plan,
    pack_int_lists,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.srt.disaggregation.utils import (
    build_transfer_entry_pairs,
    compute_mamba_state_slice_byte_blocks,
    resolve_dcp_dst_entry_indices,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# --------------------------------------------------------------------------
# Real Kimi-K3 topology (from the HF config's text_config).
# --------------------------------------------------------------------------
NUM_LAYERS = 93
# HF linear_attn_config.full_attn_layers is 1-based: [4, 8, ..., 92, 93].
FULL_ATTN_LAYERS = [l - 1 for l in list(range(4, 93, 4)) + [93]]
KDA_LAYERS = sorted(set(range(NUM_LAYERS)) - set(FULL_ATTN_LAYERS))
NUM_LINEAR_HEADS = 96

PREFILL_TP = 1
PREFILL_PP = 16
DECODE_TP = 16
DCP_SIZE = 16


def pp_stage_bounds(pp_rank: int):
    return get_pp_indices(NUM_LAYERS, pp_rank, PREFILL_PP)


def stage_full_attn_layers(pp_rank: int):
    start, end = pp_stage_bounds(pp_rank)
    return [l for l in FULL_ATTN_LAYERS if start <= l < end]


def stage_kda_layers(pp_rank: int):
    start, end = pp_stage_bounds(pp_rank)
    return [l for l in KDA_LAYERS if start <= l < end]


class TestKimiK3PP16StagePartition(CustomTestCase):
    def test_stage_partition_covers_all_layers_exactly_once(self):
        seen = []
        for r in range(PREFILL_PP):
            start, end = pp_stage_bounds(r)
            seen.extend(range(start, end))
        self.assertEqual(seen, list(range(NUM_LAYERS)))

    def test_every_stage_has_mla_and_kda_layers(self):
        # Non-empty per-stage MLA kv_layer_ids and KDA state_layer_ids are what
        # let the PP16 sender pair its entries against the PP1 decode. An
        # all-KDA (or all-MLA) stage would exercise different code paths, so
        # pin that the real 93-layer / PP16 split never produces one.
        for r in range(PREFILL_PP):
            self.assertTrue(
                stage_full_attn_layers(r), f"stage {r} has no MLA layer"
            )
            self.assertTrue(stage_kda_layers(r), f"stage {r} has no KDA layer")

    def test_stage_layer_ids_union_matches_decode_lists(self):
        mla_union, kda_union = [], []
        for r in range(PREFILL_PP):
            mla_union.extend(stage_full_attn_layers(r))
            kda_union.extend(stage_kda_layers(r))
        self.assertEqual(mla_union, FULL_ATTN_LAYERS)
        self.assertEqual(kda_union, KDA_LAYERS)


def make_decode_manager(engine_rank: int) -> CommonKVManager:
    """Decode-side manager attributes needed by _resolve_rank_mapping."""
    mgr = object.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(engine_rank=engine_rank, page_size=64)
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = True
    mgr.attn_tp_size = DECODE_TP
    mgr.attn_cp_size = 1
    mgr.attn_cp_rank = 0
    mgr.pp_size = 1
    mgr.pp_rank = 0
    mgr.dcp_size = DCP_SIZE
    mgr.enable_all_cp_ranks_for_transfer = False
    return mgr


def make_prefill_manager(
    pp_rank: int,
    *,
    page_size: int = 64,
    kv_item_lens=None,
    kv_data_ptrs=None,
    kv_layer_ids=None,
) -> MooncakeKVManager:
    """Prefill-side (one PP stage) manager for the transfer-path methods."""
    mgr = object.__new__(MooncakeKVManager)
    stage_layers = stage_full_attn_layers(pp_rank)
    mgr.kv_args = SimpleNamespace(
        engine_rank=0,
        page_size=page_size,
        kv_data_ptrs=(
            kv_data_ptrs
            if kv_data_ptrs is not None
            else [10_000_000 * (i + 1) for i in range(len(stage_layers))]
        ),
        kv_item_lens=(
            kv_item_lens
            if kv_item_lens is not None
            else [576 * page_size] * len(stage_layers)
        ),
        kv_layer_ids=(
            kv_layer_ids if kv_layer_ids is not None else list(stage_layers)
        ),
        prefill_start_layer=pp_stage_bounds(pp_rank)[0],
    )
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = True
    mgr.attn_tp_size = PREFILL_TP
    mgr.attn_cp_size = 1
    mgr.attn_cp_rank = 0
    mgr.pp_size = PREFILL_PP
    mgr.pp_rank = pp_rank
    mgr.dcp_size = 1
    mgr.dcp_rank = 0
    mgr.enable_custom_mem_pool = False
    return mgr


class TestDecodeRankMappingTP16ToTP1PP16(CustomTestCase):
    def _info(self):
        return PrefillServerInfo(
            attn_tp_size=PREFILL_TP,
            attn_cp_size=1,
            dp_size=1,
            pp_size=PREFILL_PP,
            page_size=64,
            kv_cache_dtype=None,
            follow_bootstrap_room=False,
        )

    def test_every_decode_rank_targets_prefill_tp0_and_all_pp_stages(self):
        for engine_rank in range(DECODE_TP):
            mgr = make_decode_manager(engine_rank)
            info = self._info()
            mgr._resolve_rank_mapping(info)
            # MLA KV is unsharded on the TP1 prefill: every decode rank pulls
            # from prefill TP rank 0 ...
            self.assertEqual(info.target_tp_rank, 0)
            self.assertEqual(info.target_tp_ranks, [0])
            # ... and from every one of the 16 PP stages.
            self.assertEqual(info.target_pp_ranks, list(range(PREFILL_PP)))
            self.assertEqual(info.target_cp_ranks, [0])
            # The prefill waits for all 16 decode-rank registrations per room.
            self.assertEqual(info.required_dst_info_num, DECODE_TP)
            # Each decode rank waits for one response per PP stage.
            self.assertEqual(info.required_prefill_response_num, PREFILL_PP)


class TestPrefillDcpRelayoutGate(CustomTestCase):
    def test_prefill_dcp1_to_decode_dcp16_requires_relayout(self):
        mgr = make_prefill_manager(0)
        for dst_rank in range(DCP_SIZE):
            self.assertTrue(mgr.requires_dcp_relayout(DCP_SIZE, dst_rank))

    def test_matching_dcp_sizes_must_pair_same_rank(self):
        mgr = make_prefill_manager(0)
        mgr.dcp_size = DCP_SIZE
        mgr.dcp_rank = 3
        self.assertFalse(mgr.requires_dcp_relayout(DCP_SIZE, 3))
        with self.assertRaisesRegex(RuntimeError, "matching DCP ranks"):
            mgr.requires_dcp_relayout(DCP_SIZE, 4)

    def test_non_mla_pool_rejects_dcp_fanout(self):
        mgr = make_prefill_manager(0)
        mgr.is_hybrid_mla_backend = False
        mgr.is_mla_backend = False
        with self.assertRaisesRegex(RuntimeError, "Unsupported PD DCP topology"):
            mgr.requires_dcp_relayout(DCP_SIZE, 0)

    def test_prepare_dcp_token_item_lens_requires_matching_geometry(self):
        page_size = 64
        mgr = make_prefill_manager(5, page_size=page_size)
        n = len(mgr.kv_args.kv_item_lens)
        token_lens = mgr.prepare_dcp_token_item_lens([576 * page_size] * n)
        self.assertEqual(token_lens, [576] * n)
        with self.assertRaisesRegex(RuntimeError, "geometry differs"):
            mgr.prepare_dcp_token_item_lens([128 * page_size] * n)


class TestDcpEntryPairingUnderPP16(CustomTestCase):
    def test_each_stage_resolves_decode_entry_positions(self):
        for r in range(PREFILL_PP):
            stage_layers = stage_full_attn_layers(r)
            dst_indices = resolve_dcp_dst_entry_indices(
                stage_layers,
                FULL_ATTN_LAYERS,
                len(stage_layers),
                len(FULL_ATTN_LAYERS),
            )
            self.assertEqual(
                dst_indices,
                [FULL_ATTN_LAYERS.index(l) for l in stage_layers],
            )

    def test_missing_decode_layer_entry_fails_loudly(self):
        stage_layers = stage_full_attn_layers(7)
        truncated = [l for l in FULL_ATTN_LAYERS if l not in stage_layers]
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            resolve_dcp_dst_entry_indices(
                stage_layers, truncated, len(stage_layers), len(truncated)
            )

    def test_one_sided_layer_metadata_is_rejected_under_pp(self):
        # Pins the current, deliberate behavior behind the "spec decode off"
        # rule of this shape: a decode peer that registers without layer ids
        # (e.g. running a DSPARK draft KV pool) cannot pair with a PP-sharded
        # prefill, and must fail loudly instead of transferring wrong layers.
        stage_layers = stage_full_attn_layers(7)
        with self.assertRaisesRegex(RuntimeError, "both PD peers or neither"):
            build_transfer_entry_pairs(
                stage_layers,
                [],
                len(stage_layers),
                len(FULL_ATTN_LAYERS),
                allow_positional_fallback=False,
            )

    def test_kda_state_pairing_with_two_tensors_per_layer(self):
        # MambaPool registers conv + temporal state buffers, so per-layer ids
        # repeat across the tensor groups: [conv l0..lk, temporal l0..lk].
        stage = stage_kda_layers(4)
        src_ids = stage + stage
        dst_ids = KDA_LAYERS + KDA_LAYERS
        pairs = build_transfer_entry_pairs(
            src_ids,
            dst_ids,
            len(src_ids),
            len(dst_ids),
            allow_positional_fallback=False,
        )
        n_stage, n_full = len(stage), len(KDA_LAYERS)
        expected = [(i, KDA_LAYERS.index(l)) for i, l in enumerate(stage)] + [
            (n_stage + i, n_full + KDA_LAYERS.index(l))
            for i, l in enumerate(stage)
        ]
        self.assertEqual(pairs, expected)


class TestDcpTokenPlan(CustomTestCase):
    def _brute_force_owned(
        self, num_tokens, dcp_rank, chunk_start_global, page_size
    ):
        """Global chunk-relative offsets owned by dcp_rank."""
        return [
            o
            for o in range(num_tokens)
            if (chunk_start_global + o) % DCP_SIZE == dcp_rank
        ]

    def test_chunked_prefill_union_covers_each_token_exactly_once(self):
        # Two 16384-token prefill chunks plus a ragged tail, page size 64:
        # exactly what the deployed shape produces with
        # --chunked-prefill-size 16384.
        page_size = 64
        chunk_tokens = 16384
        total_tokens = 2 * chunk_tokens + 511
        decode_prefix_len = page_size * DCP_SIZE * 3  # virtual-page aligned
        send_tokens = total_tokens - decode_prefix_len

        rng = np.random.default_rng(7)
        total_pages = -(-send_tokens // page_size)
        src_pages_all = rng.permutation(total_pages * 2)[:total_pages].astype(
            np.int32
        )

        seen = {}  # chunk-region-relative global offset -> (rank, dst_local)
        for dcp_rank in range(DCP_SIZE):
            # Per-rank shard: ceil-divide, page-granular allocation.
            rank_tokens = -(-send_tokens // DCP_SIZE)
            rank_pages = -(-rank_tokens // page_size)
            dst_pages = (np.arange(rank_pages, dtype=np.int32) * 5 + dcp_rank)
            covered = 0
            page_cursor = 0
            while covered < send_tokens:
                n = min(chunk_tokens, send_tokens - covered)
                n_pages = -(-n // page_size)
                plan = build_dcp_token_transfer_plan(
                    src_pages_all[page_cursor : page_cursor + n_pages],
                    dst_pages,
                    physical_page_size=page_size,
                    dcp_size=DCP_SIZE,
                    dcp_rank=dcp_rank,
                    src_page_offset=page_cursor,
                    decode_prefix_len=decode_prefix_len,
                    num_kv_tokens=n,
                )
                chunk_start_global = decode_prefix_len + page_cursor * page_size
                owned = self._brute_force_owned(
                    n, dcp_rank, chunk_start_global, page_size
                )
                self.assertEqual(len(plan.src_token_indices), len(owned))
                for src_tok, dst_tok, o in zip(
                    plan.src_token_indices, plan.dst_token_indices, owned
                ):
                    # Source: the o-th token of this chunk.
                    local = page_cursor * page_size + o - page_cursor * page_size
                    self.assertEqual(
                        src_tok,
                        src_pages_all[page_cursor + o // page_size] * page_size
                        + o % page_size,
                    )
                    # Destination: this rank's shard, prefix excluded.
                    rel = page_cursor * page_size + o
                    dst_local = rel // DCP_SIZE
                    self.assertEqual(
                        dst_tok,
                        dst_pages[dst_local // page_size] * page_size
                        + dst_local % page_size,
                    )
                    key = rel
                    self.assertNotIn(key, seen)
                    seen[key] = dcp_rank
                covered += n
                page_cursor += n_pages
        # Union over all 16 ranks covers every sent token exactly once.
        self.assertEqual(len(seen), send_tokens)

    def test_misaligned_decode_prefix_len_is_rejected(self):
        page_size = 64
        with self.assertRaisesRegex(ValueError, "virtual"):
            build_dcp_token_transfer_plan(
                np.arange(4, dtype=np.int32),
                np.arange(4, dtype=np.int32),
                physical_page_size=page_size,
                dcp_size=DCP_SIZE,
                dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=page_size,  # aligned to page, not to page*16
                num_kv_tokens=page_size * 4,
            )

    def test_insufficient_destination_pages_is_rejected(self):
        page_size = 64
        with self.assertRaisesRegex(ValueError, "Insufficient destination"):
            build_dcp_token_transfer_plan(
                np.arange(32, dtype=np.int32),
                np.arange(1, dtype=np.int32),
                physical_page_size=page_size,
                dcp_size=DCP_SIZE,
                dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=page_size * 32,
            )


class RecordingEngine:
    def __init__(self):
        self.blocks = []

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        self.blocks.extend(zip(src_addrs, dst_addrs, lengths))
        return 0


class TestMooncakeSendKvcacheDcpPP16(CustomTestCase):
    def test_pp_stage_send_is_byte_exact_for_every_dcp_rank(self):
        # One PP stage sending one chunk to a DCP16 decode rank, verified
        # byte-for-byte: layer entries must land on the decode entry of the
        # same global layer id, tokens must land on the rank's shard slots.
        page_size = 64
        token_len = 576  # fp8 MLA KV: kv_lora_rank 512 + qk_rope 64
        pp_rank = 6
        stage_layers = stage_full_attn_layers(pp_rank)
        mgr = make_prefill_manager(
            pp_rank,
            page_size=page_size,
            kv_item_lens=[token_len * page_size] * len(stage_layers),
        )
        mgr.engine = RecordingEngine()

        dst_kv_ptrs = [1_000_000_000 * (j + 1) for j in range(len(FULL_ATTN_LAYERS))]
        token_item_lens = [token_len] * len(stage_layers)

        num_tokens = 3 * page_size + 17
        src_pages = np.array([9, 2, 5, 11], dtype=np.int32)
        dst_pages = np.array([4, 40, 8, 1], dtype=np.int32)
        src_page_offset = 4  # a later chunk of the same request
        decode_prefix_len = page_size * DCP_SIZE  # one virtual page

        for dcp_rank in range(DCP_SIZE):
            mgr.engine = RecordingEngine()
            ret = mgr.send_kvcache_dcp(
                "session",
                src_pages,
                list(dst_kv_ptrs),
                dst_pages,
                dcp_token_item_lens=token_item_lens,
                dst_dcp_size=DCP_SIZE,
                dst_dcp_rank=dcp_rank,
                src_page_offset=src_page_offset,
                decode_prefix_len=decode_prefix_len,
                num_kv_tokens=num_tokens,
                executor=None,
                dst_layer_ids=list(FULL_ATTN_LAYERS),
            )
            self.assertEqual(ret, 0)

            # Expand coalesced blocks back into per-token transfers.
            transfers = set()
            for src_addr, dst_addr, length in mgr.engine.blocks:
                self.assertEqual(length % token_len, 0)
                for t in range(length // token_len):
                    transfers.add(
                        (src_addr + t * token_len, dst_addr + t * token_len)
                    )

            # Brute-force reference.
            expected = set()
            chunk_start = decode_prefix_len + src_page_offset * page_size
            for i, lid in enumerate(stage_layers):
                src_base = mgr.kv_args.kv_data_ptrs[i]
                dst_base = dst_kv_ptrs[FULL_ATTN_LAYERS.index(lid)]
                for o in range(num_tokens):
                    if (chunk_start + o) % DCP_SIZE != dcp_rank:
                        continue
                    src_tok = (
                        int(src_pages[o // page_size]) * page_size
                        + o % page_size
                    )
                    rel = src_page_offset * page_size + o
                    dst_local = rel // DCP_SIZE
                    dst_tok = (
                        int(dst_pages[dst_local // page_size]) * page_size
                        + dst_local % page_size
                    )
                    expected.add(
                        (
                            src_base + src_tok * token_len,
                            dst_base + dst_tok * token_len,
                        )
                    )
            self.assertEqual(transfers, expected)

    def test_decode_missing_stage_layer_fails_instead_of_mistransfer(self):
        pp_rank = 6
        stage_layers = stage_full_attn_layers(pp_rank)
        mgr = make_prefill_manager(pp_rank)
        mgr.engine = RecordingEngine()
        bad_dst_layers = [l for l in FULL_ATTN_LAYERS if l != stage_layers[0]]
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            mgr.send_kvcache_dcp(
                "session",
                np.array([0], dtype=np.int32),
                [1] * len(bad_dst_layers),
                np.array([0], dtype=np.int32),
                dcp_token_item_lens=[576] * len(stage_layers),
                dst_dcp_size=DCP_SIZE,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=1,
                executor=None,
                dst_layer_ids=bad_dst_layers,
            )


class TestKdaStateSliceTP1ToTP16(CustomTestCase):
    def test_temporal_state_slices_tile_the_full_head_dim(self):
        # temporal_state dim: 96 linear heads on the TP1 prefill,
        # 6 heads per decode rank at TP16.
        src_dim = NUM_LINEAR_HEADS
        dst_dim = NUM_LINEAR_HEADS // DECODE_TP
        head_state_bytes = 128 * 128 * 4  # head_dim x state_size, fp32
        src_item_len = src_dim * head_state_bytes
        dst_item_len = dst_dim * head_state_bytes

        covered = []
        for dst_rank in range(DECODE_TP):
            blocks = compute_mamba_state_slice_byte_blocks(
                src_item_len=src_item_len,
                dst_item_len=dst_item_len,
                src_dim=src_dim,
                dst_dim=dst_dim,
                outer_count=1,
                src_attn_tp_size=PREFILL_TP,
                dst_attn_tp_size=DECODE_TP,
                dst_tp_rank_in_group=dst_rank,
                local_tp_rank_in_group=0,
                conv_shard_groups=None,
            )
            self.assertEqual(len(blocks), 1)
            src_off, dst_off, nbytes = blocks[0]
            self.assertEqual(dst_off, 0)
            self.assertEqual(nbytes, dst_item_len)
            covered.append((src_off, src_off + nbytes))
        covered.sort()
        # The 16 ranks' slices tile [0, src_item_len) with no gap or overlap.
        self.assertEqual(covered[0][0], 0)
        self.assertEqual(covered[-1][1], src_item_len)
        for (_, prev_end), (next_start, _) in zip(covered, covered[1:]):
            self.assertEqual(prev_end, next_start)

    def test_conv_state_qkv_shards_and_rows_tile_exactly(self):
        # Kimi KDA conv state: [q | k | v] concat, each sub-block head-sharded
        # independently, with outer_count = conv_kernel-1 = 3 rows per slot.
        head_dim = 128
        q_dim = k_dim = NUM_LINEAR_HEADS * head_dim
        v_dim = NUM_LINEAR_HEADS * head_dim
        conv_shard_groups = [q_dim, k_dim, v_dim]
        src_dim = sum(conv_shard_groups) // PREFILL_TP
        dst_dim = sum(conv_shard_groups) // DECODE_TP
        outer_count = 3
        bytes_per_dim = 2  # bf16
        src_item_len = outer_count * src_dim * bytes_per_dim
        dst_item_len = outer_count * dst_dim * bytes_per_dim

        all_src_bytes = set()
        for dst_rank in range(DECODE_TP):
            blocks = compute_mamba_state_slice_byte_blocks(
                src_item_len=src_item_len,
                dst_item_len=dst_item_len,
                src_dim=src_dim,
                dst_dim=dst_dim,
                outer_count=outer_count,
                src_attn_tp_size=PREFILL_TP,
                dst_attn_tp_size=DECODE_TP,
                dst_tp_rank_in_group=dst_rank,
                local_tp_rank_in_group=0,
                conv_shard_groups=conv_shard_groups,
            )
            # One block per (row, sub-block).
            self.assertEqual(len(blocks), outer_count * len(conv_shard_groups))
            dst_covered = set()
            for src_off, dst_off, nbytes in blocks:
                src_rng = set(range(src_off, src_off + nbytes))
                dst_rng = set(range(dst_off, dst_off + nbytes))
                self.assertFalse(src_rng & all_src_bytes)
                self.assertFalse(dst_rng & dst_covered)
                all_src_bytes |= src_rng
                dst_covered |= dst_rng
            # Each rank's dst buffer is fully written.
            self.assertEqual(len(dst_covered), dst_item_len)
        # All ranks together consume the full src state exactly once.
        self.assertEqual(len(all_src_bytes), src_item_len)


class TestMooncakeDcpRegistrationWire(CustomTestCase):
    def test_decode_registration_roundtrip_carries_dcp_and_layer_ids(self):
        # Mirrors MooncakeKVReceiver._register_kv_args's packing order for the
        # DCP16 decode role, then drives the prefill-side bootstrap gating.
        kv_ptrs = [1_000_000 * (j + 1) for j in range(len(FULL_ATTN_LAYERS))]
        state_ptrs = [[7_000_000 + j for j in range(2 * len(KDA_LAYERS))]]
        state_item_lens = [[4096] * (2 * len(KDA_LAYERS))]
        state_dims = [[6] * (2 * len(KDA_LAYERS))]
        state_layer_ids = [KDA_LAYERS + KDA_LAYERS]
        page_size = 64
        kv_item_len = 576 * page_size
        msg = [
            b"None",
            b"10.0.0.8",
            b"31000",
            b"session-9",
            b"".join(struct.pack("Q", p) for p in kv_ptrs),
            b"".join(struct.pack("Q", p) for p in [123, 456]),
            pack_int_lists(state_ptrs, "Q"),
            b"9",  # dst_tp_rank
            str(DECODE_TP).encode("ascii"),
            str(kv_item_len).encode("ascii"),
            pack_int_lists(state_item_lens, "I"),
            pack_int_lists(state_dims, "I"),
            b"".join(struct.pack("I", l) for l in FULL_ATTN_LAYERS),
            pack_int_lists(state_layer_ids, "I"),
            b"",  # staging base ptr
            b"",  # staging total size
            str(DCP_SIZE).encode("ascii"),
            b"9",  # dst_dcp_rank
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_dcp_size, DCP_SIZE)
        self.assertEqual(info.dst_dcp_rank, 9)
        self.assertEqual(info.dst_attn_tp_size, DECODE_TP)
        self.assertEqual(info.dst_kv_layer_ids, FULL_ATTN_LAYERS)
        self.assertEqual(info.dst_state_layer_ids, state_layer_ids)
        self.assertEqual(info.dst_kv_ptrs, kv_ptrs)
        self.assertEqual(info.dst_kv_item_len, kv_item_len)

        # Prefill PP-stage bootstrap gating on this registration.
        mgr = make_prefill_manager(
            3, page_size=page_size, kv_item_lens=None
        )
        mgr.kv_args.kv_item_lens = [kv_item_len] * len(
            stage_full_attn_layers(3)
        )
        self.assertTrue(
            mgr.requires_dcp_relayout(info.dst_dcp_size, info.dst_dcp_rank)
        )
        token_lens = mgr.prepare_dcp_token_item_lens(
            [info.dst_kv_item_len] * len(mgr.kv_args.kv_item_lens)
        )
        self.assertEqual(token_lens, [576] * len(mgr.kv_args.kv_item_lens))


class FakeServerArgs(SimpleNamespace):
    def override(self, tag, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestPPMambaBudgetCharge(CustomTestCase):
    """Pins the KIMI_K3_PP_MAMBA_BUDGET_FIX carried from the deployed image."""

    def _make_configurator(self, *, pp_rank, pp_size, per_req_bytes):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        cfg = object.__new__(KVCacheConfigurator)
        start, end = (
            get_pp_indices(NUM_LAYERS, pp_rank, pp_size)
            if pp_size > 1
            else (0, NUM_LAYERS)
        )
        cfg.layer_info = SimpleNamespace(start_layer=start, end_layer=end)
        cfg.mambaish_config = SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                mamba_cache_per_req=per_req_bytes,
                layers=list(KDA_LAYERS),
                replayssm_ring_bytes_per_req=lambda **kwargs: 0,
            )
        )
        cfg.server_args = FakeServerArgs(
            pp_size=pp_size,
            enable_linear_replayssm_spec=False,
            linear_replayssm_cache_len=0,
            max_mamba_cache_size=None,
            disable_radix_cache=False,
            mamba_full_memory_ratio=1.0,
            max_running_requests=None,
            speculative_num_draft_tokens=None,
        )
        cfg.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        cfg.hybrid_gdn_config = None
        cfg.model_config = SimpleNamespace(hf_config=None)
        cfg.ps = SimpleNamespace(attn_dp_size=1)
        return cfg

    def test_pp16_budget_charge_uses_max_stage_kda_share(self):
        per_req = 470 << 20  # ~470MB full-model KDA state per request
        rest_gb = 52.0
        max_stage_kda = max(len(stage_kda_layers(r)) for r in range(PREFILL_PP))
        self.assertLess(max_stage_kda, len(KDA_LAYERS))

        def fake_all_reduce(tensor, op=None, group=None):
            # World-group MAX over per-stage KDA layer counts.
            tensor.fill_(max_stage_kda)

        cfg = self._make_configurator(
            pp_rank=0, pp_size=PREFILL_PP, per_req_bytes=per_req
        )
        with patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_world_group",
            return_value=SimpleNamespace(cpu_group=None),
        ), patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            remaining = cfg._handle_max_mamba_cache(rest_gb)

        share = max_stage_kda / len(KDA_LAYERS)
        per_slot = max(int(per_req * share), 1)
        budget_bytes = rest_gb * 0.5 * (1 << 30)  # ratio 1.0 -> half the rest
        expected_slots = int((budget_bytes - per_slot) // per_slot)
        self.assertEqual(cfg.server_args.max_mamba_cache_size, expected_slots)
        expected_remaining = rest_gb - (expected_slots + 1) * per_slot / (1 << 30)
        self.assertAlmostEqual(remaining, expected_remaining, places=6)
        # The PP16 solve must land far above the ~120-slot regression the fix
        # was written for (full-model charge), and match the per-stage share.
        full_charge_slots = int(
            (budget_bytes - per_req) // per_req
        )
        self.assertGreater(expected_slots, full_charge_slots * 8)

    def test_pp1_behavior_is_unchanged_and_never_all_reduces(self):
        per_req = 470 << 20
        rest_gb = 52.0
        cfg = self._make_configurator(pp_rank=0, pp_size=1, per_req_bytes=per_req)

        def must_not_be_called(*args, **kwargs):
            raise AssertionError("pp_size == 1 must not touch all_reduce")

        with patch(
            "torch.distributed.all_reduce", side_effect=must_not_be_called
        ):
            remaining = cfg._handle_max_mamba_cache(rest_gb)

        budget_bytes = rest_gb * 0.5 * (1 << 30)
        expected_slots = int((budget_bytes - per_req) // per_req)
        self.assertEqual(cfg.server_args.max_mamba_cache_size, expected_slots)
        expected_remaining = rest_gb - (expected_slots + 1) * per_req / (1 << 30)
        self.assertAlmostEqual(remaining, expected_remaining, places=6)


if __name__ == "__main__":
    unittest.main()
