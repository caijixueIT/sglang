import types
import unittest

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
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
        self.assertEqual(merged.tolist(), [14, 15, 16, 17])


if __name__ == "__main__":
    unittest.main()
