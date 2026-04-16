from __future__ import annotations

import json
import os
import shutil
import unittest
from unittest.mock import patch

from chatmock.app import create_app
from chatmock.upstream import start_upstream_request


class FakeResponse:
    def __init__(self, lines: list[bytes], content: bytes = b"") -> None:
        self._lines = lines
        self.content = content
        self.status_code = 200
        self.headers = {}
        self.encoding = "utf-8"
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line.decode("utf-8") if decode_unicode else line

    def close(self) -> None:
        self.closed = True


class VerboseDumpTests(unittest.TestCase):
    def test_verbose_upstream_request_writes_timestamped_request_and_response_files(self) -> None:
        response = FakeResponse(
            [
                b'data: {"type":"response.output_text.delta","delta":"hello"}',
                b"data: [DONE]",
            ]
        )
        input_items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]

        tmp = os.path.join(os.getcwd(), "tests", ".tmp-verbose-dump-test")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        try:
            app = create_app(verbose=True)
            with (
                patch.dict(os.environ, {"CHATGPT_LOCAL_HOME": tmp}),
                patch("chatmock.upstream.get_effective_chatgpt_auth", return_value=("token", "account")),
                patch("chatmock.upstream.requests.post", return_value=response),
                patch("builtins.print"),
                app.test_request_context("/v1/chat/completions", method="POST"),
            ):
                upstream, error_resp = start_upstream_request("gpt-5.4", input_items, instructions="inst")

                self.assertIsNone(error_resp)
                self.assertEqual(
                    list(upstream.iter_lines(decode_unicode=False)),
                    [
                        b'data: {"type":"response.output_text.delta","delta":"hello"}',
                        b"data: [DONE]",
                    ],
                )
                upstream.close()

            dump_dir = os.path.join(tmp, "verbose-dumps")
            request_files = sorted(name for name in os.listdir(dump_dir) if name.endswith("-request.json"))
            response_files = sorted(name for name in os.listdir(dump_dir) if name.endswith("-response.sse"))

            self.assertEqual(len(request_files), 1)
            self.assertEqual(len(response_files), 1)
            self.assertRegex(request_files[0], r"^\d{8}-\d{6}-\d{6}-\d+-request\.json$")
            self.assertRegex(response_files[0], r"^\d{8}-\d{6}-\d{6}-\d+-response\.sse$")

            with open(os.path.join(dump_dir, request_files[0]), "r", encoding="utf-8") as fp:
                request_dump = json.load(fp)
            self.assertEqual(request_dump["method"], "POST")
            self.assertEqual(request_dump["json"]["model"], "gpt-5.4")
            self.assertEqual(request_dump["json"]["instructions"], "inst")
            self.assertEqual(request_dump["json"]["input"], input_items)

            with open(os.path.join(dump_dir, response_files[0]), "rb") as fp:
                response_dump = fp.read()
            self.assertIn(b'"delta":"hello"', response_dump)
            self.assertIn(b"data: [DONE]", response_dump)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
