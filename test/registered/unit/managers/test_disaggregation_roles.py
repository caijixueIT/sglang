import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVArgs, KVTransferDirection
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.server_args import ServerArgs


class _DummyCommonKVManager(CommonKVManager):
    def register_to_bootstrap(self):
        pass


class TestDisaggregationRoles(unittest.TestCase):
    def _make_kv_args(self, direction: str) -> KVArgs:
        kv_args = KVArgs()
        kv_args.system_dp_rank = 0
        kv_args.pp_rank = 0
        kv_args.page_size = 1
        kv_args.transfer_direction = direction
        return kv_args

    @patch("sglang.srt.disaggregation.common.conn.get_pp_group", return_value=None)
    @patch(
        "sglang.srt.disaggregation.common.conn.get_zmq_socket_on_host",
        return_value=(12345, object()),
    )
    @patch(
        "sglang.srt.disaggregation.common.conn.get_local_ip_auto",
        return_value="127.0.0.1",
    )
    @patch("sglang.srt.disaggregation.common.conn.get_attention_dp_rank", return_value=0)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_dp_size", return_value=1)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_cp_rank", return_value=0)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_cp_size", return_value=1)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_tp_rank", return_value=0)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_tp_size", return_value=1)
    def test_role_mapping_matches_transfer_direction(self, *_mocks):
        prefill_args = ServerArgs(model_path="dummy", disaggregation_mode="prefill")
        decode_args = ServerArgs(model_path="dummy", disaggregation_mode="decode")

        sender_prefill = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.PREFILL_TO_DECODE.value),
            DisaggregationMode.PREFILL,
            prefill_args,
        )
        receiver_decode = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.PREFILL_TO_DECODE.value),
            DisaggregationMode.DECODE,
            decode_args,
        )
        sender_decode = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.DECODE_TO_PREFILL.value),
            DisaggregationMode.DECODE,
            decode_args,
        )
        receiver_prefill = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.DECODE_TO_PREFILL.value),
            DisaggregationMode.PREFILL,
            prefill_args,
        )

        self.assertTrue(sender_prefill.is_sender_role)
        self.assertFalse(sender_prefill.is_receiver_role)
        self.assertTrue(receiver_decode.is_receiver_role)
        self.assertFalse(receiver_decode.is_sender_role)
        self.assertTrue(sender_decode.is_sender_role)
        self.assertFalse(sender_decode.is_receiver_role)
        self.assertTrue(receiver_prefill.is_receiver_role)
        self.assertFalse(receiver_prefill.is_sender_role)

    @patch("sglang.srt.disaggregation.common.conn.get_pp_group", return_value=None)
    @patch(
        "sglang.srt.disaggregation.common.conn.get_zmq_socket_on_host",
        return_value=(12345, object()),
    )
    @patch(
        "sglang.srt.disaggregation.common.conn.get_local_ip_auto",
        return_value="127.0.0.1",
    )
    @patch("sglang.srt.disaggregation.common.conn.get_attention_dp_rank", return_value=0)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_dp_size", return_value=1)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_cp_rank", return_value=0)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_cp_size", return_value=1)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_tp_rank", return_value=0)
    @patch("sglang.srt.disaggregation.common.conn.get_attention_tp_size", return_value=1)
    def test_reverse_transfer_uses_dualpath_bootstrap_port(self, *_mocks):
        prefill_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            dualpath_enable=True,
            dualpath_decode_bootstrap_port=19001,
        )
        decode_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            dualpath_enable=True,
            dualpath_decode_bootstrap_port=19001,
        )

        sender_decode = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.DECODE_TO_PREFILL.value),
            DisaggregationMode.DECODE,
            decode_args,
        )
        receiver_prefill = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.DECODE_TO_PREFILL.value),
            DisaggregationMode.PREFILL,
            prefill_args,
        )
        sender_prefill = _DummyCommonKVManager(
            self._make_kv_args(KVTransferDirection.PREFILL_TO_DECODE.value),
            DisaggregationMode.PREFILL,
            prefill_args,
        )

        self.assertEqual(sender_decode.bootstrap_port, 19001)
        self.assertEqual(receiver_prefill.bootstrap_port, 19001)
        self.assertEqual(
            sender_prefill.bootstrap_port, prefill_args.disaggregation_bootstrap_port
        )

    @patch("sglang.srt.managers.disagg_service.get_kv_class")
    def test_start_disagg_service_starts_decode_bootstrap_for_dualpath(
        self, mock_get_kv_class
    ):
        bootstrap_server = object()
        bootstrap_ctor = MagicMock(return_value=bootstrap_server)
        mock_get_kv_class.return_value = bootstrap_ctor

        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            dualpath_enable=True,
        )
        result = start_disagg_service(server_args)

        self.assertIs(result, bootstrap_server)
        bootstrap_ctor.assert_called_once_with(host=server_args.host, port=8999)

    @patch("sglang.srt.managers.disagg_service.get_kv_class")
    def test_start_disagg_service_honors_explicit_dualpath_bootstrap_port(
        self, mock_get_kv_class
    ):
        bootstrap_server = object()
        bootstrap_ctor = MagicMock(return_value=bootstrap_server)
        mock_get_kv_class.return_value = bootstrap_ctor

        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            dualpath_enable=True,
            dualpath_decode_bootstrap_port=19001,
        )
        result = start_disagg_service(server_args)

        self.assertIs(result, bootstrap_server)
        bootstrap_ctor.assert_called_once_with(host=server_args.host, port=19001)

    @patch("sglang.srt.managers.disagg_service.get_kv_class")
    def test_start_disagg_service_skips_decode_bootstrap_without_dualpath(
        self, mock_get_kv_class
    ):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        result = start_disagg_service(server_args)

        self.assertIsNone(result)
        mock_get_kv_class.assert_not_called()
