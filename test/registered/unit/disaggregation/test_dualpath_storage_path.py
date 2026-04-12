import socket
import time
import types
import unittest
from queue import Queue
from unittest import mock

import requests
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    _resolve_reverse_host_pool,
    _resolve_reverse_prefill_start_layer,
)
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
    DecodeStorageReadSession,
)
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.entrypoints.openai.protocol import CachedTokensDetails
from sglang.srt.entrypoints.openai.utils import process_cached_tokens_details_from_ret
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    NSATokenToKVPoolHost,
)

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestDualPathStoragePath(unittest.TestCase):
    class _FakeReverseSender:
        def __init__(self, mgr=None, bootstrap_room=None, **kwargs):
            del kwargs
            self.kv_mgr = mgr if mgr is not None else types.SimpleNamespace(transfer_infos=None)
            self.bootstrap_room = bootstrap_room

        def init(self, *_args, **_kwargs):
            return None

        def poll(self):
            return KVPoll.WaitingForInput

        def send(self, *_args, **_kwargs):
            return None

        def clear(self):
            transfer_infos = getattr(self.kv_mgr, "transfer_infos", None)
            if transfer_infos is not None and self.bootstrap_room is not None:
                transfer_infos.pop(self.bootstrap_room, None)

    class _FakePtrTensor:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, idx):
            return types.SimpleNamespace(item=lambda: self._values[idx])

    def test_process_cached_tokens_details_keeps_storage_path(self):
        req = types.SimpleNamespace(return_cached_tokens_details=True)
        ret_item = {
            "meta_info": {
                "cached_tokens_details": {
                    "device": 16,
                    "host": 0,
                    "storage": 8,
                    "storage_backend": "mooncake",
                    "storage_path": "decode",
                }
            }
        }

        details = process_cached_tokens_details_from_ret(ret_item, req)

        self.assertIsInstance(details, CachedTokensDetails)
        self.assertEqual(details.storage_path, "decode")
        self.assertEqual(details.storage_backend, "mooncake")

    def test_output_processor_keeps_decode_storage_details_without_local_backend(self):
        scheduler = types.SimpleNamespace(
            enable_hicache_storage=False,
            server_args=types.SimpleNamespace(hicache_storage_backend="file"),
            _get_storage_backend_type=lambda: "file",
        )
        req = types.SimpleNamespace(
            rid="req-output",
            cached_tokens=8,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=8,
            cached_tokens_storage_path="decode",
        )

        details = SchedulerOutputProcessorMixin._get_cached_tokens_details(
            scheduler, req
        )

        self.assertEqual(
            details,
            {
                "device": 0,
                "host": 0,
                "storage": 8,
                "storage_backend": "file",
                "storage_path": "decode",
            },
        )

    def test_bootstrap_server_stores_dualpath_read_info(self):
        port = _get_free_port()
        server = CommonKVBootstrapServer("127.0.0.1", port)
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    response = requests.get(f"{base_url}/health", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    time.sleep(0.05)
            else:
                self.fail("bootstrap server did not become ready in time")

            response = requests.post(
                f"{base_url}/register_dualpath_read_info",
                json={
                    "bootstrap_room": 123,
                    "matched_len": 64,
                    "last_hash": "hash-64",
                    "prefix_keys": ["hash-16", "hash-32", "hash-48"],
                },
                timeout=2,
            )
            self.assertEqual(response.status_code, 200)

            query_response = requests.post(
                f"{base_url}/query_dualpath_read_info",
                json={"bootstrap_rooms": [123, 456]},
                timeout=2,
            )
            self.assertEqual(query_response.status_code, 200)
            payload = query_response.json()

            self.assertEqual(
                payload["123"],
                {
                    "matched_len": 64,
                    "last_hash": "hash-64",
                    "prefix_keys": ["hash-16", "hash-32", "hash-48"],
                },
            )
            self.assertNotIn("456", payload)
        finally:
            server.close()

    def test_bootstrap_server_stores_dualpath_read_status(self):
        port = _get_free_port()
        server = CommonKVBootstrapServer("127.0.0.1", port)
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    response = requests.get(f"{base_url}/health", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    time.sleep(0.05)
            else:
                self.fail("bootstrap server did not become ready in time")

            response = requests.post(
                f"{base_url}/register_dualpath_read_info",
                json={
                    "bootstrap_room": 321,
                    "matched_len": 64,
                    "last_hash": "hash-64",
                    "prefix_keys": ["hash-16"],
                },
                timeout=2,
            )
            self.assertEqual(response.status_code, 200)

            status_response = requests.post(
                f"{base_url}/register_dualpath_read_status",
                json={
                    "bootstrap_room": 321,
                    "decode_storage_read_started": True,
                    "decode_storage_read_completed": True,
                    "decode_storage_read_hit_tokens": 48,
                    "decode_reverse_transfer_ready": True,
                    "decode_reverse_transfer_completed": True,
                },
                timeout=2,
            )
            self.assertEqual(status_response.status_code, 200)

            query_response = requests.post(
                f"{base_url}/query_dualpath_read_info",
                json={"bootstrap_rooms": [321]},
                timeout=2,
            )
            self.assertEqual(query_response.status_code, 200)
            payload = query_response.json()

            self.assertTrue(payload["321"]["decode_storage_read_started"])
            self.assertTrue(payload["321"]["decode_storage_read_completed"])
            self.assertEqual(payload["321"]["decode_storage_read_hit_tokens"], 48)
            self.assertTrue(payload["321"]["decode_reverse_transfer_ready"])
            self.assertTrue(payload["321"]["decode_reverse_transfer_completed"])
            self.assertEqual(payload["321"]["matched_len"], 64)
        finally:
            server.close()

    def test_reverse_host_pool_prefers_decode_offload_pool(self):
        decode_host_pool = object()
        tree_host_pool = object()

        resolved = _resolve_reverse_host_pool(
            types.SimpleNamespace(token_to_kv_pool_host=tree_host_pool),
            types.SimpleNamespace(decode_host_mem_pool=decode_host_pool),
        )

        self.assertIs(resolved, decode_host_pool)

    def test_process_reverse_transfers_starts_storage_read_before_reverse_guard(self):
        queue = object.__new__(DecodePreallocQueue)
        queue.scheduler = types.SimpleNamespace(decode_offload_manager=object())
        queue.reverse_kv_manager = None
        queue.reverse_host_pool = None

        called = {"value": False}

        def mark_called():
            called["value"] = True

        queue._maybe_start_dualpath_storage_reads = mark_called

        queue.process_reverse_transfers()

        self.assertTrue(called["value"])

    def test_reverse_prefill_start_layer_defaults_to_zero(self):
        self.assertEqual(
            _resolve_reverse_prefill_start_layer(types.SimpleNamespace()), 0
        )

    def test_reverse_kv_manager_sets_prefill_start_layer(self):
        queue = object.__new__(DecodePreallocQueue)
        queue.tp_rank = 0
        queue.pp_rank = 0
        queue.is_mla_backend = False
        queue.transfer_backend = "fake-backend"
        queue.metadata_buffers = types.SimpleNamespace(
            get_buf_infos=lambda: ([], [], [])
        )
        queue.reverse_host_pool = types.SimpleNamespace(
            start_layer=7,
            page_size=16,
            head_num=8,
            get_contiguous_buf_infos=lambda: ([11, 22], [128, 128], [16, 16]),
        )
        queue.scheduler = types.SimpleNamespace(
            dp_rank=0,
            gpu_id=1,
            server_args=types.SimpleNamespace(
                disaggregation_ib_device=None,
                dualpath_ib_traffic_class=None,
            ),
            model_config=types.SimpleNamespace(get_total_num_kv_heads=lambda: 8),
        )

        captured = {}

        class FakeKVArgs:
            pass

        def fake_get_kv_class(_backend, class_type):
            if class_type.name == "KVARGS":
                return FakeKVArgs

            def fake_manager_class(kv_args, *_args):
                captured["kv_args"] = kv_args
                return types.SimpleNamespace(kv_args=kv_args)

            return fake_manager_class

        with mock.patch(
            "sglang.srt.disaggregation.decode.get_kv_class",
            side_effect=fake_get_kv_class,
        ), mock.patch(
            "sglang.srt.disaggregation.decode.get_attention_tp_size",
            return_value=1,
        ):
            queue._init_reverse_kv_manager()

        self.assertEqual(captured["kv_args"].prefill_start_layer, 7)

    def test_prefill_stage_ready_waits_for_decode_completion(self):
        queue = object.__new__(PrefillBootstrapQueue)
        queue.scheduler = types.SimpleNamespace(
            page_size=16,
            server_args=types.SimpleNamespace(
                dualpath_prefill_reverse_wait_budget_s=0.2
            )
        )
        req = types.SimpleNamespace(
            dualpath_selected_path="de_read",
            dualpath_decode_storage_read_completed=False,
            dualpath_decode_storage_read_requested=False,
            bootstrap_room=11,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            origin_input_ids=list(range(32)),
            dualpath_prefill_matched_len=0,
            dualpath_prefill_last_hash="hash-16",
            dualpath_prefill_prefix_keys=None,
            dualpath_decode_storage_read_started=False,
            dualpath_decode_storage_read_hit_tokens=0,
            dualpath_decode_reverse_transfer_ready=False,
            dualpath_decode_reverse_transfer_completed=False,
        )

        with mock.patch(
            "sglang.srt.disaggregation.prefill.CommonKVReceiver.query_dualpath_read_infos",
            return_value={
                "11": {
                    "matched_len": 0,
                    "last_hash": "hash-16",
                    "decode_storage_read_started": True,
                    "decode_storage_read_completed": False,
                }
            },
        ) as query_mock:
            ready = queue.is_dualpath_reverse_stage_ready(req)

        self.assertFalse(ready)
        self.assertTrue(req.dualpath_decode_storage_read_started)
        self.assertFalse(req.dualpath_decode_storage_read_completed)
        query_mock.assert_called_once_with("127.0.0.1:31500", [11])

    def test_prefill_stage_ready_allows_decode_completion_without_reverse_ready(self):
        queue = object.__new__(PrefillBootstrapQueue)
        queue.scheduler = types.SimpleNamespace(
            page_size=16,
            server_args=types.SimpleNamespace(
                dualpath_prefill_reverse_wait_budget_s=0.2
            )
        )
        req = types.SimpleNamespace(
            dualpath_selected_path="de_read",
            dualpath_decode_storage_read_completed=False,
            dualpath_decode_storage_read_requested=False,
            dualpath_decode_storage_read_started=False,
            bootstrap_room=12,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            origin_input_ids=list(range(32)),
            dualpath_prefill_matched_len=0,
            dualpath_prefill_last_hash="hash-16",
            dualpath_prefill_prefix_keys=None,
            dualpath_decode_storage_read_hit_tokens=0,
            dualpath_decode_reverse_transfer_ready=False,
            dualpath_decode_reverse_transfer_completed=False,
        )

        with mock.patch(
            "sglang.srt.disaggregation.prefill.CommonKVReceiver.query_dualpath_read_infos",
            return_value={
                "12": {
                    "matched_len": 0,
                    "last_hash": "hash-16",
                    "decode_storage_read_started": True,
                    "decode_storage_read_completed": True,
                    "decode_storage_read_hit_tokens": 16,
                    "decode_reverse_transfer_ready": False,
                }
            },
        ):
            ready = queue.is_dualpath_reverse_stage_ready(req)

        self.assertTrue(ready)
        self.assertTrue(req.dualpath_decode_storage_read_completed)
        self.assertEqual(req.dualpath_decode_storage_read_hit_tokens, 16)
        self.assertFalse(req.dualpath_decode_reverse_transfer_ready)

    def test_prefill_wait_budget_requires_decode_completion_and_hits(self):
        queue = object.__new__(PrefillBootstrapQueue)
        queue.scheduler = types.SimpleNamespace(
            page_size=16,
            server_args=types.SimpleNamespace(
                dualpath_prefill_reverse_wait_budget_s=0.2
            )
        )
        req = types.SimpleNamespace(
            dualpath_selected_path="de_read",
            dualpath_decode_storage_read_completed=False,
            dualpath_decode_storage_read_requested=False,
            dualpath_decode_storage_read_started=False,
            bootstrap_room=12,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            origin_input_ids=list(range(32)),
            dualpath_prefill_matched_len=0,
            dualpath_prefill_last_hash="hash-16",
            dualpath_prefill_prefix_keys=None,
            dualpath_decode_storage_read_hit_tokens=0,
            dualpath_decode_reverse_transfer_ready=False,
            dualpath_decode_reverse_transfer_completed=False,
        )

        with mock.patch(
            "sglang.srt.disaggregation.prefill.CommonKVReceiver.query_dualpath_read_infos",
            return_value={
                "12": {
                    "matched_len": 0,
                    "last_hash": "hash-16",
                    "decode_storage_read_started": True,
                    "decode_storage_read_completed": True,
                    "decode_storage_read_hit_tokens": 16,
                    "decode_reverse_transfer_ready": False,
                }
            },
        ):
            wait_budget_s = queue.get_dualpath_reverse_wait_budget_s(req)

        self.assertEqual(wait_budget_s, 0.2)
        self.assertTrue(req.dualpath_decode_storage_read_started)
        self.assertTrue(req.dualpath_decode_storage_read_completed)
        self.assertEqual(req.dualpath_decode_storage_read_hit_tokens, 16)
        self.assertFalse(req.dualpath_decode_reverse_transfer_ready)

    def test_prefill_wait_budget_skips_when_decode_completion_has_no_hits(self):
        queue = object.__new__(PrefillBootstrapQueue)
        queue.scheduler = types.SimpleNamespace(
            page_size=16,
            server_args=types.SimpleNamespace(
                dualpath_prefill_reverse_wait_budget_s=0.2
            ),
        )
        req = types.SimpleNamespace(
            dualpath_selected_path="de_read",
            dualpath_decode_storage_read_completed=False,
            dualpath_decode_storage_read_requested=False,
            dualpath_decode_storage_read_started=False,
            bootstrap_room=13,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            origin_input_ids=list(range(32)),
            dualpath_prefill_matched_len=0,
            dualpath_prefill_last_hash="hash-16",
            dualpath_prefill_prefix_keys=None,
            dualpath_decode_storage_read_hit_tokens=0,
            dualpath_decode_reverse_transfer_ready=False,
            dualpath_decode_reverse_transfer_completed=False,
        )

        with mock.patch(
            "sglang.srt.disaggregation.prefill.CommonKVReceiver.query_dualpath_read_infos",
            return_value={
                "13": {
                    "matched_len": 0,
                    "last_hash": "hash-16",
                    "decode_storage_read_started": True,
                    "decode_storage_read_completed": True,
                    "decode_storage_read_hit_tokens": 0,
                }
            },
        ):
            wait_budget_s = queue.get_dualpath_reverse_wait_budget_s(req)

        self.assertEqual(wait_budget_s, 0.0)

    def test_decode_storage_read_start_registers_bootstrap_status(self):
        queue = object.__new__(DecodePreallocQueue)
        req = types.SimpleNamespace(
            rid="req-1",
            dualpath_selected_path="de_read",
            dualpath_decode_storage_read_requested=False,
            dualpath_decode_storage_read_started=False,
            dualpath_reverse_transfer_done=False,
            bootstrap_room=99,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            origin_input_ids=list(range(32)),
        )
        queue.queue = [types.SimpleNamespace(req=req)]
        queue.pending_reqs = []
        queue.transfer_queue = types.SimpleNamespace(queue=[])
        queue.scheduler = types.SimpleNamespace(
            decode_offload_manager=types.SimpleNamespace(
                page_size=16,
                start_storage_read=lambda *_args, **_kwargs: True,
            )
        )

        with mock.patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_dualpath_read_infos",
            return_value={
                "99": {
                    "matched_len": 0,
                    "last_hash": "hash-16",
                    "prefix_keys": ["hash-0"],
                }
            },
        ), mock.patch(
            "sglang.srt.disaggregation.decode.CommonKVManager.register_dualpath_read_status"
        ) as register_mock:
            queue._maybe_start_dualpath_storage_reads()

        self.assertTrue(req.dualpath_decode_storage_read_requested)
        self.assertTrue(req.dualpath_decode_storage_read_started)
        register_mock.assert_called_once_with(
            "127.0.0.1:31500",
            99,
            decode_storage_read_started=True,
        )

    def test_process_reverse_transfers_reads_waiting_queue_requests(self):
        queue = object.__new__(DecodePreallocQueue)
        req = types.SimpleNamespace(
            rid="req-waiting",
            dualpath_selected_path="de_read",
            dualpath_reverse_sender=None,
            dualpath_reverse_transfer_done=False,
            dualpath_decode_bootstrap_host="127.0.0.1",
            dualpath_decode_bootstrap_port=31501,
            bootstrap_room=21,
            dualpath_reverse_sender_aux_index=-1,
            dualpath_reverse_source_indices=None,
            dualpath_reverse_source_tokens=0,
            dualpath_reverse_send_started=False,
            dualpath_decode_reverse_transfer_ready=False,
            dualpath_decode_reverse_transfer_completed=False,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            cached_tokens_storage_path=None,
        )
        queue.queue = []
        queue.pending_reqs = []
        queue.transfer_queue = types.SimpleNamespace(queue=[])
        queue.reverse_transfer_queue = []
        queue.reverse_kv_manager = types.SimpleNamespace(
            transfer_infos={
                21: {
                    "session-1": types.SimpleNamespace(
                        dst_kv_indices=[10],
                        is_dummy=False,
                    )
                }
            }
        )
        queue.reverse_host_pool = types.SimpleNamespace(page_size=4)
        queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(2)
        queue.metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((2, 16), dtype=torch.int32)
        )
        queue.tp_rank = 0
        queue.pp_rank = 0
        queue.bootstrap_port = 31501
        queue.transfer_backend = "fake-backend"
        queue._maybe_start_dualpath_storage_reads = lambda: None
        queue._dualpath_sender_has_metadata = lambda _req: True
        queue.scheduler = types.SimpleNamespace(
            decode_offload_manager=types.SimpleNamespace(
                take_completed_storage_read=lambda rid: (
                    torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                    4,
                    time.time(),
                )
                if rid == "req-waiting"
                else None
            ),
            waiting_queue=[req],
            running_batch=types.SimpleNamespace(reqs=[]),
            last_batch=types.SimpleNamespace(reqs=[]),
        )

        with mock.patch(
            "sglang.srt.disaggregation.decode.get_kv_class",
            return_value=self._FakeReverseSender,
        ), mock.patch(
            "sglang.srt.disaggregation.decode.CommonKVManager.register_dualpath_read_status"
        ) as register_mock:
            queue.process_reverse_transfers()

        self.assertIsNotNone(req.dualpath_reverse_sender)
        self.assertEqual(len(queue.reverse_transfer_queue), 1)
        self.assertTrue(req.dualpath_reverse_send_started)
        self.assertEqual(req.cached_tokens, 4)
        self.assertEqual(req.cached_tokens_storage, 4)
        self.assertEqual(req.cached_tokens_storage_path, "decode")
        register_mock.assert_called_once_with(
            "127.0.0.1:31500",
            21,
            decode_storage_read_started=True,
            decode_storage_read_completed=True,
            decode_storage_read_hit_tokens=4,
            decode_reverse_transfer_ready=True,
        )

    def test_process_reverse_transfers_streams_remaining_pages_across_chunks(self):
        queue = object.__new__(DecodePreallocQueue)
        req = types.SimpleNamespace(
            rid="req-stream",
            dualpath_selected_path="de_read",
            dualpath_reverse_sender=None,
            dualpath_reverse_transfer_done=False,
            dualpath_decode_bootstrap_host="127.0.0.1",
            dualpath_decode_bootstrap_port=31501,
            bootstrap_room=31,
            dualpath_reverse_sender_aux_index=-1,
            dualpath_reverse_source_indices=None,
            dualpath_reverse_source_tokens=0,
            dualpath_reverse_source_page_offset=0,
            dualpath_reverse_source_token_offset=0,
            dualpath_reverse_current_chunk_pages=0,
            dualpath_reverse_current_chunk_tokens=0,
            dualpath_reverse_send_started=False,
            dualpath_decode_reverse_transfer_ready=False,
            dualpath_decode_reverse_transfer_completed=False,
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            cached_tokens_storage_path=None,
        )
        queue.queue = []
        queue.pending_reqs = []
        queue.transfer_queue = types.SimpleNamespace(queue=[])
        queue.reverse_transfer_queue = []
        queue.reverse_host_pool = types.SimpleNamespace(page_size=4, free=mock.Mock())
        queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(4)
        queue.metadata_buffers = types.SimpleNamespace(
            cached_tokens=torch.zeros((4, 16), dtype=torch.int32)
        )
        queue.tp_rank = 0
        queue.pp_rank = 0
        queue.bootstrap_port = 31501
        queue.transfer_backend = "fake-backend"
        queue._maybe_start_dualpath_storage_reads = lambda: None

        queue.reverse_kv_manager = types.SimpleNamespace(
            transfer_infos={
                31: {
                    "session-1": types.SimpleNamespace(
                        dst_kv_indices=[100, 101],
                        is_dummy=False,
                    )
                }
            },
            required_prefill_response_num_table={},
            prefill_response_tracker={},
        )

        reads = {"done": False}

        def take_completed_storage_read(rid):
            if rid != "req-stream" or reads["done"]:
                return None
            reads["done"] = True
            return torch.tensor(list(range(12)), dtype=torch.int32), 12, time.time()

        queue.scheduler = types.SimpleNamespace(
            decode_offload_manager=types.SimpleNamespace(
                take_completed_storage_read=take_completed_storage_read
            ),
            waiting_queue=[req],
            running_batch=types.SimpleNamespace(reqs=[]),
            last_batch=types.SimpleNamespace(reqs=[]),
        )

        sent_chunks = []

        class ChunkedReverseSender:
            def __init__(self, mgr=None, bootstrap_room=None, **kwargs):
                del kwargs
                self.kv_mgr = mgr
                self.bootstrap_room = bootstrap_room
                self._sent = None

            def init(self, num_kv_indices, aux_index=None):
                self.num_kv_indices = num_kv_indices
                self.aux_index = aux_index

            def poll(self):
                return KVPoll.Success if self._sent is not None else KVPoll.WaitingForInput

            def send(self, kv_indices, *_args, **_kwargs):
                self._sent = list(kv_indices)
                sent_chunks.append(self._sent)

            def clear(self):
                self.kv_mgr.transfer_infos.pop(self.bootstrap_room, None)

        with mock.patch(
            "sglang.srt.disaggregation.decode.get_kv_class",
            return_value=ChunkedReverseSender,
        ), mock.patch(
            "sglang.srt.disaggregation.decode.CommonKVManager.register_dualpath_read_status"
        ) as register_mock:
            queue.process_reverse_transfers()
            self.assertEqual(sent_chunks, [[0, 1]])
            self.assertEqual(len(queue.reverse_transfer_queue), 1)
            self.assertEqual(req.dualpath_reverse_source_page_offset, 0)
            self.assertFalse(req.dualpath_decode_reverse_transfer_completed)

            queue.process_reverse_transfers()
            self.assertEqual(req.dualpath_reverse_source_page_offset, 2)
            self.assertEqual(req.dualpath_reverse_source_token_offset, 8)
            self.assertFalse(req.dualpath_decode_reverse_transfer_completed)
            self.assertEqual(len(queue.reverse_transfer_queue), 1)

            queue.reverse_kv_manager.transfer_infos = {
                31: {
                    "session-1": types.SimpleNamespace(
                        dst_kv_indices=[102],
                        is_dummy=False,
                    )
                }
            }

            queue.process_reverse_transfers()
            self.assertEqual(sent_chunks, [[0, 1], [2]])
            self.assertEqual(len(queue.reverse_transfer_queue), 1)

            queue.process_reverse_transfers()

        self.assertTrue(req.dualpath_decode_reverse_transfer_completed)
        self.assertTrue(req.dualpath_reverse_transfer_done)
        self.assertIsNone(req.dualpath_reverse_source_indices)
        self.assertEqual(req.dualpath_reverse_source_tokens, 0)
        self.assertEqual(len(queue.reverse_transfer_queue), 0)
        queue.reverse_host_pool.free.assert_called_once()
        register_mock.assert_has_calls(
            [
                mock.call(
                    "127.0.0.1:31500",
                    31,
                    decode_storage_read_started=True,
                    decode_storage_read_completed=True,
                    decode_storage_read_hit_tokens=8,
                    decode_reverse_transfer_ready=True,
                ),
                mock.call(
                    "127.0.0.1:31500",
                    31,
                    decode_storage_read_started=True,
                    decode_storage_read_completed=True,
                    decode_storage_read_hit_tokens=4,
                    decode_reverse_transfer_ready=True,
                ),
                mock.call(
                    "127.0.0.1:31500",
                    31,
                    decode_storage_read_started=True,
                    decode_storage_read_completed=True,
                    decode_storage_read_hit_tokens=12,
                    decode_reverse_transfer_ready=True,
                    decode_reverse_transfer_completed=True,
                ),
            ]
        )

    def test_commit_transfer_preserves_decode_storage_details_when_metadata_empty(self):
        queue = object.__new__(DecodeTransferQueue)
        queue.metadata_buffers = types.SimpleNamespace(
            get_buf=lambda _idx: (
                torch.tensor([7], dtype=torch.int32),
                torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32),
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros((1, 1), dtype=torch.float32),
                torch.tensor([17], dtype=torch.int64),
            )
        )
        queue.scheduler = types.SimpleNamespace(
            server_args=types.SimpleNamespace(disaggregation_transfer_backend="mooncake")
        )
        queue.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
        req = types.SimpleNamespace(
            rid="req-commit",
            bootstrap_host="127.0.0.1",
            bootstrap_port=31500,
            bootstrap_room=17,
            output_ids=[],
            cached_tokens=4,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=4,
            cached_tokens_storage_path="decode",
            return_logprob=False,
            top_logprobs_num=0,
            time_stats=types.SimpleNamespace(set_wait_queue_entry_time=lambda: None),
        )
        decode_req = types.SimpleNamespace(
            metadata_buffer_index=0,
            req=req,
            kv_receiver=types.SimpleNamespace(clear=lambda: None),
        )

        should_remove = queue._commit_transfer_to_req(decode_req)

        self.assertTrue(should_remove)
        self.assertEqual(req.output_ids, [7])
        self.assertEqual(req.cached_tokens, 4)
        self.assertEqual(req.cached_tokens_storage, 4)
        self.assertEqual(req.cached_tokens_storage_path, "decode")

    def test_decode_storage_read_completion_registers_bootstrap_status(self):
        manager = object.__new__(DecodeKVCacheOffloadManager)
        host_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        operation = types.SimpleNamespace(
            hash_value=["h1"],
            completed_tokens=4,
            host_indices=host_indices,
            is_terminated=lambda: False,
        )
        manager.ongoing_storage_reads = {
            "req-1": DecodeStorageReadSession(
                req_id="req-1",
                host_indices=host_indices,
                operation=operation,
                started_at=1.0,
                requested_tokens=4,
                bootstrap_addr="127.0.0.1:31500",
                bootstrap_room=7,
            )
        }
        manager.completed_storage_reads = {}
        manager.storage_read_hit_tokens_total = 0
        manager.page_size = 4
        manager.decode_host_mem_pool = types.SimpleNamespace(free=mock.Mock())

        with mock.patch(
            "sglang.srt.disaggregation.decode_kvcache_offload_manager.CommonKVManager.register_dualpath_read_status"
        ) as register_mock:
            manager._check_storage_read_progress()

        self.assertEqual(manager.storage_read_hit_tokens_total, 4)
        self.assertIn("req-1", manager.completed_storage_reads)
        register_mock.assert_called_once_with(
            "127.0.0.1:31500",
            7,
            decode_storage_read_started=True,
            decode_storage_read_completed=True,
            decode_storage_read_hit_tokens=4,
        )

    def test_revoke_prefetch_marks_operation_terminated(self):
        controller = object.__new__(HiCacheController)
        controller.prefetch_revoke_queue = Queue()
        controller.host_mem_release_queue = Queue()
        controller.mem_pool_host = types.SimpleNamespace(page_size=2)

        operation = PrefetchOperation(
            request_id="req-1",
            host_indices=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            token_ids=[10, 11, 12, 13],
        )

        controller.revoke_prefetch(operation, storage_hit_count=0)

        self.assertTrue(operation.is_terminated())
        self.assertEqual(controller.prefetch_revoke_queue.get_nowait(), "req-1")
        self.assertTrue(
            torch.equal(
                controller.host_mem_release_queue.get_nowait(),
                torch.tensor([1, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                controller.host_mem_release_queue.get_nowait(),
                torch.tensor([3, 4], dtype=torch.int32),
            )
        )
        self.assertTrue(controller.host_mem_release_queue.empty())

    def test_mha_host_pool_exposes_contiguous_buffer_infos(self):
        pool = object.__new__(MHATokenToKVPoolHost)
        pool.layer_num = 2
        pool.token_stride_size = 128
        pool.k_data_refs = [types.SimpleNamespace(nbytes=1024), types.SimpleNamespace(nbytes=2048)]
        pool.v_data_refs = [types.SimpleNamespace(nbytes=4096), types.SimpleNamespace(nbytes=8192)]
        pool.k_data_ptrs = self._FakePtrTensor([11, 22])
        pool.v_data_ptrs = self._FakePtrTensor([33, 44])

        data_ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()

        self.assertEqual(data_ptrs, [11, 22, 33, 44])
        self.assertEqual(data_lens, [1024, 2048, 4096, 8192])
        self.assertEqual(item_lens, [128, 128, 128, 128])

    def test_mla_host_pool_exposes_contiguous_buffer_infos(self):
        pool = object.__new__(MLATokenToKVPoolHost)
        pool.layer_num = 2
        pool.token_stride_size = 96
        pool.kv_buffer = [types.SimpleNamespace(nbytes=512), types.SimpleNamespace(nbytes=768)]
        pool.data_ptrs = self._FakePtrTensor([101, 202])

        data_ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()

        self.assertEqual(data_ptrs, [101, 202])
        self.assertEqual(data_lens, [512, 768])
        self.assertEqual(item_lens, [96, 96])

    def test_nsa_host_pool_inherits_contiguous_buffer_infos(self):
        pool = object.__new__(NSATokenToKVPoolHost)
        pool.layer_num = 1
        pool.token_stride_size = 80
        pool.kv_buffer = [types.SimpleNamespace(nbytes=640)]
        pool.data_ptrs = self._FakePtrTensor([303])

        data_ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()

        self.assertEqual(data_ptrs, [303])
        self.assertEqual(data_lens, [640])
        self.assertEqual(item_lens, [80])


if __name__ == "__main__":
    unittest.main()
