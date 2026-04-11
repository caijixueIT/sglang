import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)
from sglang.utils import wait_for_http_ready


BASE_HOST = "127.0.0.1"
LB_PORT = 31000
PREFILL_PORT = 31100
DECODE_PORT = 31200
BOOTSTRAP_PORT = 31500
MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN


def _post_completion(base_url: str, prompt: str, max_tokens: int = 48) -> dict:
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _query_loads(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/v1/loads?include=disagg", timeout=30)
    resp.raise_for_status()
    return resp.json()


def _decode_storage_read_hit_tokens(load_payload: dict) -> int:
    loads = load_payload.get("loads", [])
    if not loads:
        return 0
    return loads[0].get("disaggregation", {}).get("decode_storage_read_hit_tokens", 0)


def _launch_router(prefill_url: str, decode_url: str) -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill",
        prefill_url,
        "--decode",
        decode_url,
        "--host",
        BASE_HOST,
        "--port",
        str(LB_PORT),
        "--dualpath-enable",
        "--dualpath-mode",
        "decode_only",
    ]
    proc = subprocess.Popen(cmd)
    wait_for_http_ready(
        url=f"http://{BASE_HOST}:{LB_PORT}/health",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        process=proc,
    )
    return proc


def main():
    prefill_url = f"http://{BASE_HOST}:{PREFILL_PORT}"
    decode_url = f"http://{BASE_HOST}:{DECODE_PORT}"
    lb_url = f"http://{BASE_HOST}:{LB_PORT}"
    hicache_dir = tempfile.mkdtemp(prefix="dualpath-hicache-")
    env = os.environ.copy()
    env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = hicache_dir
    env["SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE"] = "16"

    processes = []
    try:
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            str(BOOTSTRAP_PORT),
            "--tp",
            "1",
            "--page-size",
            "16",
            "--enable-hierarchical-cache",
            "--hicache-storage-backend",
            "file",
            "--hicache-ratio",
            "2",
            "--dualpath-enable",
        ]
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            str(BOOTSTRAP_PORT),
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--page-size",
            "16",
            "--hicache-ratio",
            "2",
            "--hicache-storage-backend",
            "file",
            "--disaggregation-decode-enable-offload-kvcache",
            "--num-reserved-decode-tokens",
            "128",
            "--dualpath-enable",
        ]

        process_prefill = popen_launch_pd_server(
            MODEL,
            prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=env,
        )
        processes.append(process_prefill)
        process_decode = popen_launch_pd_server(
            MODEL,
            decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=env,
        )
        processes.append(process_decode)

        wait_for_http_ready(
            url=f"{prefill_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=process_prefill,
        )
        wait_for_http_ready(
            url=f"{decode_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=process_decode,
        )

        process_lb = _launch_router(prefill_url, decode_url)
        processes.append(process_lb)

        prompt = "DualPath pressure test prompt. " * 256

        # Round 1: populate decode-side offload/storage.
        for _ in range(8):
            _post_completion(lb_url, prompt)
        time.sleep(10)

        before = _query_loads(decode_url)
        before_hits = _decode_storage_read_hit_tokens(before)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: _post_completion(lb_url, prompt), range(16)))
        elapsed = time.perf_counter() - start

        after = _query_loads(decode_url)
        after_hits = _decode_storage_read_hit_tokens(after)

        completed = sum(1 for item in results if item.get("choices"))
        summary = {
            "model": MODEL,
            "completed_requests": completed,
            "elapsed_s": round(elapsed, 3),
            "qps": round(completed / elapsed, 3) if elapsed > 0 else 0.0,
            "decode_storage_read_hit_tokens_before": before_hits,
            "decode_storage_read_hit_tokens_after": after_hits,
            "decode_storage_read_hit_tokens_delta": after_hits - before_hits,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))

        if completed != 16:
            raise RuntimeError(f"Expected 16 completed requests, got {completed}")
        if after_hits <= before_hits:
            raise RuntimeError(
                "Decode-side storage read metric did not increase during pressure test"
            )
    finally:
        for proc in reversed(processes):
            if proc is not None:
                try:
                    kill_process_tree(proc.pid)
                except Exception:
                    pass
        shutil.rmtree(hicache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
