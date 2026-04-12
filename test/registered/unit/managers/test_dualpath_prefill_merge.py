import types
import unittest

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    _refresh_cached_token_breakdown,
)
from sglang.srt.sampling.sampling_params import SamplingParams


class _StubReverseReceiver:
    def __init__(self, metadata_buffers, hit_tokens: int):
        self.metadata_buffers = metadata_buffers
        self.hit_tokens = hit_tokens

    def send_metadata(self, kv_indices, aux_index=None, state_indices=None):
        del kv_indices, state_indices
        self.metadata_buffers.cached_tokens[aux_index][0] = self.hit_tokens

    def poll(self):
        return KVPoll.Success

    def clear(self):
        pass


class _DelayedStartReverseReceiver:
    def __init__(self, metadata_buffers, hit_tokens: int, req, clock):
        self.metadata_buffers = metadata_buffers
        self.hit_tokens = hit_tokens
        self.req = req
        self.clock = clock

    def send_metadata(self, kv_indices, aux_index=None, state_indices=None):
        del kv_indices, state_indices
        self.metadata_buffers.cached_tokens[aux_index][0] = self.hit_tokens

    def poll(self):
        if (
            self.req.dualpath_decode_reverse_transfer_ready
            and self.clock["t"] >= 0.015
        ):
            return KVPoll.Success
        return KVPoll.WaitingForInput

    def clear(self):
        pass


class TestDualPathPrefillMerge(unittest.TestCase):
    def test_reverse_merge_shrinks_prefill_work(self):
        req = Req(
            rid="rid-1",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=7,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.set_extend_input_len(len(req.fill_ids))

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.05,
                create_reverse_receiver=lambda _req: _StubReverseReceiver(
                    metadata_buffers, hit_tokens=4
                )
            ),
        )

        out_cache_loc = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64)
        merged = batch._apply_dualpath_reverse_prefill_merge(
            scheduler, out_cache_loc, [0]
        )

        self.assertEqual(req.prefix_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.host_hit_length, 4)
        self.assertEqual(req.storage_hit_length, 4)
        self.assertEqual(req.cached_tokens_device, 0)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 4)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        self.assertEqual(merged.tolist(), [14, 15, 16, 17])

    def test_reverse_merge_appends_tail_after_partial_prefix_hit(self):
        req = Req(
            rid="rid-2",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=8,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.prefix_indices = torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        req.host_hit_length = 0
        req.set_extend_input_len(4)

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[101, 102, 103, 104, 201, 202, 203, 204]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.05,
                create_reverse_receiver=lambda _req: _StubReverseReceiver(
                    metadata_buffers, hit_tokens=4
                )
            ),
        )

        out_cache_loc = torch.tensor([201, 202, 203, 204], dtype=torch.int64)
        merged = batch._apply_dualpath_reverse_prefill_merge(
            scheduler, out_cache_loc, [0]
        )

        self.assertEqual(
            req.prefix_indices.tolist(), [101, 102, 103, 104, 201, 202, 203]
        )
        self.assertEqual(req.extend_input_len, 1)
        self.assertEqual(req.host_hit_length, 3)
        self.assertEqual(req.storage_hit_length, 3)
        self.assertEqual(req.cached_tokens_device, 4)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 3)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        self.assertEqual(merged.tolist(), [204])

    def test_reverse_merge_respects_logprob_prefix_limit(self):
        req = Req(
            rid="rid-2b",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=82,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
            return_logprob=True,
        )
        req.logprob_start_len = 5
        req.fill_ids = list(req.origin_input_ids)
        req.prefix_indices = torch.tensor([101, 102, 103, 104], dtype=torch.int64)
        req.host_hit_length = 0
        req.set_extend_input_len(4)

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[101, 102, 103, 104, 201, 202, 203, 204]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.05,
                create_reverse_receiver=lambda _req: _StubReverseReceiver(
                    metadata_buffers, hit_tokens=4
                ),
            ),
        )

        out_cache_loc = torch.tensor([201, 202, 203, 204], dtype=torch.int64)
        merged = batch._apply_dualpath_reverse_prefill_merge(
            scheduler, out_cache_loc, [0]
        )

        self.assertEqual(req.prefix_indices.tolist(), [101, 102, 103, 104, 201])
        self.assertEqual(req.extend_input_len, 3)
        self.assertEqual(req.host_hit_length, 1)
        self.assertEqual(req.storage_hit_length, 1)
        self.assertEqual(req.cached_tokens_device, 4)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 1)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        self.assertEqual(merged.tolist(), [202, 203, 204])

    def test_refresh_breakdown_excludes_same_request_chunk_prefix(self):
        req = Req(
            rid="rid-chunked",
            origin_input_text="",
            origin_input_ids=list(range(24499)),
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=108,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.prefix_indices = torch.arange(24498, dtype=torch.int64)
        req.already_computed = 16384
        req.cached_tokens = 8114
        req.host_hit_length = 8114
        req.storage_hit_length = 8114
        req.cached_tokens_storage_path = "decode"

        _refresh_cached_token_breakdown(req)

        self.assertEqual(req.cached_tokens_device, 0)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 8114)
        self.assertEqual(req.cached_tokens_storage_path, "decode")

    def test_reverse_merge_skips_when_wait_budget_is_zero(self):
        req = Req(
            rid="rid-3",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=9,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.set_extend_input_len(len(req.fill_ids))

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[10, 11, 12, 13]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.0,
                create_reverse_receiver=lambda _req: _StubReverseReceiver(
                    metadata_buffers, hit_tokens=4
                ),
            ),
        )

        out_cache_loc = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        merged = batch._apply_dualpath_reverse_prefill_merge(
            scheduler, out_cache_loc, [0]
        )

        self.assertEqual(req.prefix_indices.numel(), 0)
        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.storage_hit_length, 0)
        self.assertTrue(torch.equal(merged, out_cache_loc))

    def test_reverse_merge_extends_wait_after_transfer_starts(self):
        req = Req(
            rid="rid-4",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=10,
            bootstrap_host="prefill-host",
            bootstrap_port=31500,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.set_extend_input_len(len(req.fill_ids))

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        clock = {"t": 0.0}

        def refresh_status(_req):
            if clock["t"] >= 0.009:
                _req.dualpath_decode_reverse_transfer_ready = True
            return {}

        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.01,
                refresh_dualpath_reverse_status=refresh_status,
                create_reverse_receiver=lambda _req: _DelayedStartReverseReceiver(
                    metadata_buffers, hit_tokens=4, req=_req, clock=clock
                ),
            ),
        )

        out_cache_loc = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64)
        with unittest.mock.patch(
            "sglang.srt.managers.schedule_batch.time.time",
            side_effect=lambda: clock["t"],
        ), unittest.mock.patch(
            "sglang.srt.managers.schedule_batch.time.sleep",
            side_effect=lambda dt: clock.__setitem__("t", clock["t"] + dt),
        ):
            merged = batch._apply_dualpath_reverse_prefill_merge(
                scheduler, out_cache_loc, [0]
            )

        self.assertEqual(req.prefix_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.storage_hit_length, 4)
        self.assertEqual(req.cached_tokens_storage, 4)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        self.assertEqual(merged.tolist(), [14, 15, 16, 17])

    def test_reverse_merge_allows_last_chunk_of_chunked_prefill(self):
        req = Req(
            rid="rid-5",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=11,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.set_extend_input_len(len(req.fill_ids))
        req.is_chunked = 1

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
            chunked_req=None,
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.05,
                create_reverse_receiver=lambda _req: _StubReverseReceiver(
                    metadata_buffers, hit_tokens=4
                ),
            ),
        )

        out_cache_loc = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64)
        merged = batch._apply_dualpath_reverse_prefill_merge(
            scheduler, out_cache_loc, [0]
        )

        self.assertEqual(req.prefix_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.storage_hit_length, 4)
        self.assertEqual(req.cached_tokens_storage, 4)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        self.assertEqual(merged.tolist(), [14, 15, 16, 17])

    def test_reverse_merge_allows_nonfinal_chunk_of_chunked_prefill(self):
        req = Req(
            rid="rid-6",
            origin_input_text="",
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            sampling_params=SamplingParams(max_new_tokens=1),
            bootstrap_room=12,
            dualpath_decode_bootstrap_host="decode-host",
            dualpath_decode_bootstrap_port=8998,
            dualpath_selected_path="de_read",
        )
        req.fill_ids = list(req.origin_input_ids)
        req.set_extend_input_len(len(req.fill_ids))
        req.is_chunked = 1

        batch = ScheduleBatch(
            reqs=[req],
            req_to_token_pool=types.SimpleNamespace(
                req_to_token=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
            ),
            tree_cache=types.SimpleNamespace(page_size=4),
            chunked_req=req,
        )

        metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        scheduler = types.SimpleNamespace(
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(4),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=types.SimpleNamespace(
                get_dualpath_reverse_wait_budget_s=lambda _req: 0.05,
                create_reverse_receiver=lambda _req: _StubReverseReceiver(
                    metadata_buffers, hit_tokens=4
                ),
            ),
        )

        out_cache_loc = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64)
        merged = batch._apply_dualpath_reverse_prefill_merge(
            scheduler, out_cache_loc, [0]
        )

        self.assertEqual(req.prefix_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.storage_hit_length, 4)
        self.assertEqual(req.cached_tokens_storage, 4)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        self.assertEqual(merged.tolist(), [14, 15, 16, 17])


if __name__ == "__main__":
    unittest.main()
