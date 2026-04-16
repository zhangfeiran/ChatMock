from __future__ import annotations

import datetime
import json
import os
import threading
from typing import Any

from .utils import eprint, get_home_dir


def create_chatgpt_verbose_dump(method: str, url: str, payload: Any) -> "ChatGPTVerboseDump | None":
    try:
        dump_dir = os.path.join(get_home_dir(), "verbose-dumps")
        os.makedirs(dump_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        thread_id = threading.get_ident()
        for index in range(1000):
            suffix = f"-{index:03d}" if index else ""
            stem = f"{timestamp}-{thread_id}{suffix}"
            request_path = os.path.join(dump_dir, f"{stem}-request.json")
            response_path = os.path.join(dump_dir, f"{stem}-response.sse")
            try:
                with open(request_path, "x", encoding="utf-8") as fp:
                    json.dump(
                        {
                            "method": method,
                            "url": url,
                            "json": payload,
                        },
                        fp,
                        indent=2,
                        ensure_ascii=False,
                    )
                    fp.write("\n")
                with open(response_path, "xb"):
                    pass
                return ChatGPTVerboseDump(request_path=request_path, response_path=response_path)
            except FileExistsError:
                continue
        eprint("ERROR: unable to reserve ChatGPT verbose dump paths")
    except Exception as exc:
        eprint(f"ERROR: unable to create ChatGPT verbose dump: {exc}")
    return None


class ChatGPTVerboseDump:
    def __init__(self, *, request_path: str, response_path: str) -> None:
        self.request_path = request_path
        self.response_path = response_path
        self._write_error_reported = False

    def write_response_line(self, line: bytes | str) -> None:
        if isinstance(line, str):
            data = line.encode("utf-8", errors="replace")
        else:
            data = bytes(line)
        self.write_response_bytes(data + b"\n")

    def write_response_bytes(self, data: bytes) -> None:
        try:
            with open(self.response_path, "ab") as fp:
                fp.write(data)
        except Exception as exc:
            if not self._write_error_reported:
                eprint(f"ERROR: unable to write ChatGPT verbose response dump: {exc}")
                self._write_error_reported = True


class DumpingResponse:
    def __init__(self, response: Any, dump: ChatGPTVerboseDump) -> None:
        self._response = response
        self._dump = dump
        self._content_dumped = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def content(self) -> bytes:
        data = self._response.content
        if not self._content_dumped:
            self._dump.write_response_bytes(data)
            self._content_dumped = True
        return data

    @property
    def text(self) -> str:
        _ = self.content
        return self._response.text

    def iter_lines(self, *args: Any, **kwargs: Any):
        for line in self._response.iter_lines(*args, **kwargs):
            self._dump.write_response_line(line)
            yield line

    def close(self) -> None:
        self._response.close()
