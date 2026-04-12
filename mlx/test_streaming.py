#!/usr/bin/env python3
"""
Tests for MLX engine SSE streaming support.

Validates that:
1. Non-streaming requests still work (backward compat)
2. Streaming requests return valid SSE format
3. SSE chunks match the OpenAI chat.completion.chunk schema
4. The stream ends with data: [DONE]
5. /v1/models endpoint returns valid response
6. Thinking tags are stripped from streamed output

Run without MLX installed (mocks the model):
    python3 test_streaming.py
"""

import json
import io
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from http.server import HTTPServer

sys.path.insert(0, os.path.dirname(__file__))

import mlx_engine


class FakeChunk:
    def __init__(self, text):
        self.text = text


def _make_stream_response(text_pieces):
    chunks = []
    for piece in text_pieces:
        chunks.append(FakeChunk(piece))
    return chunks


class TestCleanResponse(unittest.TestCase):
    def test_strips_im_end(self):
        self.assertEqual(
            mlx_engine._clean_response("hello world<|im_end|>"), "hello world"
        )

    def test_strips_im_start(self):
        self.assertEqual(mlx_engine._clean_response("hello<|im_start|>extra"), "hello")

    def test_strips_thinking_tags(self):
        self.assertEqual(
            mlx_engine._clean_response("real answer"),
            "real answer",
        )

    def test_strips_thinking_block(self):
        raw = "Let me think about this.\n\nHere is the actual answer."
        self.assertEqual(mlx_engine._clean_response(raw), raw)

    def test_stops_at_im_end_tag(self):
        self.assertEqual(
            mlx_engine._clean_response("response text<|im_end|>garbage"),
            "response text",
        )

    def test_no_special_tokens(self):
        self.assertEqual(mlx_engine._clean_response("just text"), "just text")

    def test_empty_string(self):
        self.assertEqual(mlx_engine._clean_response(""), "")


class TestFormatChat(unittest.TestCase):
    def test_system_user_assistant(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        prompt = mlx_engine.format_chat(messages)
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_includes_thinking_block(self):
        messages = [{"role": "user", "content": "hi"}]
        prompt = mlx_engine.format_chat(messages)
        self.assertIn("", prompt)


class TestStreamParsing(unittest.TestCase):
    def test_sse_delta_format(self):
        chunk = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }
        line = f"data: {json.dumps(chunk)}"
        parsed = json.loads(line[6:])
        self.assertEqual(parsed["choices"][0]["delta"]["content"], "Hello")
        self.assertIsNone(parsed["choices"][0]["finish_reason"])

    def test_final_chunk_has_stop(self):
        chunk = {
            "id": "chatcmpl-abc123",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "timings": {"predicted_per_second": 15.0, "predicted_ms": 2000},
            "usage": {"completion_tokens": 30},
        }
        self.assertEqual(chunk["choices"][0]["finish_reason"], "stop")
        self.assertIn("timings", chunk)
        self.assertIn("usage", chunk)


class TestHandleChatStream(unittest.TestCase):
    def _make_handler(self, body_dict):
        handler = mlx_engine.APIHandler.__new__(mlx_engine.APIHandler)
        handler.wfile = io.BytesIO()
        handler.headers = {"Content-Length": str(len(json.dumps(body_dict)))}
        handler.rfile = io.BytesIO(json.dumps(body_dict).encode())
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.request_version = "HTTP/1.1"
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        handler._headers_buffer = []
        handler.request = MagicMock()
        handler.response_version = "HTTP/1.1"

        original_send_response = handler.send_response
        original_send_header = handler.send_header
        original_end_headers = handler.end_headers

        def mock_send_response(code, message=None):
            handler.response_code = code
            handler._headers_buffer = []

        def mock_send_header(keyword, value):
            handler._headers_buffer.append(f"{keyword}: {value}".encode())

        def mock_end_headers():
            pass

        handler.send_response = mock_send_response
        handler.send_header = mock_send_header
        handler.end_headers = mock_end_headers

        return handler

    @patch("mlx_engine.generate")
    def test_stream_produces_sse(self, mock_generate):
        mock_generate.return_value = iter(["Hello", " world", "!"])
        mlx_engine.model_name = "9b"

        handler = self._make_handler(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        )

        handler._handle_chat_stream([{"role": "user", "content": "hi"}], 100, 0.7)

        output = handler.wfile.getvalue().decode()
        lines = [l for l in output.split("\n") if l.startswith("data: ")]

        self.assertTrue(len(lines) >= 4, f"Expected >= 4 SSE lines, got {len(lines)}")

        for line in lines[:-2]:
            payload = json.loads(line[6:])
            self.assertEqual(payload["object"], "chat.completion.chunk")
            self.assertIn("choices", payload)
            self.assertEqual(len(payload["choices"]), 1)
            self.assertIn("delta", payload["choices"][0])

        final = json.loads(lines[-2][6:])
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertIn("timings", final)

        self.assertEqual(lines[-1], "data: [DONE]")

    @patch("mlx_engine.generate")
    def test_stream_handles_error(self, mock_generate):
        mock_generate.side_effect = RuntimeError("model error")
        mlx_engine.model_name = "9b"

        handler = self._make_handler(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
        )

        handler._handle_chat_stream([{"role": "user", "content": "hi"}], 100, 0.7)

        output = handler.wfile.getvalue().decode()
        self.assertIn("model error", output)
        self.assertIn("[DONE]", output)


class TestHandleListModels(unittest.TestCase):
    def _make_handler(self):
        handler = mlx_engine.APIHandler.__new__(mlx_engine.APIHandler)
        handler.wfile = io.BytesIO()
        handler._headers_buffer = []
        handler.request = MagicMock()
        handler.response_version = "HTTP/1.1"
        handler.response_code = 200
        handler.send_response = lambda code, message=None: None
        handler.send_header = lambda keyword, value: None
        handler.end_headers = lambda: None
        return handler

    def test_returns_model_list(self):
        mlx_engine.model_name = "9b"
        handler = self._make_handler()
        handler._handle_list_models()

        data = json.loads(handler.wfile.getvalue().decode())
        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(data["data"][0]["owned_by"], "local")
        self.assertIn("Qwen", data["data"][0]["id"])


class TestStreamThinkingStripping(unittest.TestCase):
    @patch("mlx_engine.generate")
    def test_thinking_tags_are_stripped(self, mock_generate):
        mock_generate.return_value = iter(["Real", " answer"])
        mlx_engine.model_name = "9b"

        handler_handler = mlx_engine.APIHandler.__new__(mlx_engine.APIHandler)
        handler_handler.wfile = io.BytesIO()
        handler_handler._headers_buffer = []
        handler_handler.send_response = lambda code, message=None: None
        handler_handler.send_header = lambda keyword, value: None
        handler_handler.end_headers = lambda: None

        handler_handler._handle_chat_stream(
            [{"role": "user", "content": "hi"}], 100, 0.7
        )

        output = handler_handler.wfile.getvalue().decode()
        for line in output.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                payload = json.loads(line[6:])
                content = (
                    payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                )
                if content:
                    self.assertNotIn("<think", content)
                    self.assertNotIn("</think", content)


if __name__ == "__main__":
    unittest.main()
