"""Exercise the production resolver through MCP HTTP requests and answer context.

No model weights or service credentials are required. The local model is scripted;
the real MCP SDK encodes/decodes requests through an HTTPX transport fixture.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
import os
import threading
import unittest
from unittest.mock import patch

import httpx
from mcp import ClientSession  # Import SDK type annotations before transport patching.
from rich.console import Console

import agent
import parallel_search as parallel


class MCPFixture:
    def __init__(self, failure=None):
        self.requests = []
        self.failure = failure
        self.closed = False
        self.started = threading.Event()
        self.real_client = httpx.AsyncClient

    async def handle(self, request):
        self.requests.append(request)
        data = json.loads(request.content)
        method = data['method']
        if 'id' not in data:
            return httpx.Response(202)
        if method == 'initialize':
            result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'search-fixture', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': name, 'inputSchema': {'type': 'object'}}
                                for name in ('web_search', 'web_fetch')]}
        elif method == 'tools/call':
            self.started.set()
            if self.failure == 'wait':
                try:
                    await asyncio.sleep(10)
                finally:
                    self.closed = True
            if self.failure == 'http':
                return httpx.Response(503)
            if self.failure == 'malformed':
                return httpx.Response(200, headers={'content-type': 'application/json'},
                                      content=b'not-json')
            payload = {'results': [{'url': 'https://example.org/news',
                                   'title': 'Current data',
                                   'excerpts': ['Verified event is on October 20.']}]}
            if self.failure == 'unsafe-url':
                payload['results'][0]['url'] = 'file:///etc/passwd'
            if self.failure == 'empty':
                payload['results'] = []
            result = {'content': [{'type': 'text', 'text': json.dumps(payload)}],
                      'structuredContent': payload,
                      'isError': self.failure == 'tool'}
        else:
            raise AssertionError(method)
        return httpx.Response(200, headers={'content-type': 'application/json'},
                              json={'jsonrpc': '2.0', 'id': data['id'], 'result': result})

    def client(self, **kwargs):
        return self.real_client(transport=httpx.MockTransport(self.handle), **kwargs)

    def calls(self):
        return [json.loads(r.content)['params'] for r in self.requests
                if json.loads(r.content)['method'] == 'tools/call']


class ParallelSearchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.stack.enter_context(patch.dict(os.environ, {'MAC_CODE_SEARCH_PROVIDER': 'parallel'}))
        self.fixture = MCPFixture()
        self.stack.enter_context(patch('httpx.AsyncClient', self.fixture.client))
        self.prompts = []
        def model(messages, **kwargs):
            self.prompts.append(messages)
            return 'The event is October 20 [source](https://example.org/news).', {'predicted_per_second': 12}
        self.stack.enter_context(patch.object(agent, 'llm_call', side_effect=model))

    def test_actual_resolver_request_headers_arguments_and_result_consumption(self):
        result = agent.quick_search('current event date')
        self.assertIn('October 20', result[0])
        self.assertEqual(result[1], 12)
        calls = self.fixture.calls()
        self.assertEqual([x['name'] for x in calls], ['web_search', 'web_fetch'])
        self.assertEqual(calls[0]['arguments']['objective'], 'current event date')
        self.assertEqual(calls[1]['arguments']['urls'], ['https://example.org/news'])
        self.assertEqual(calls[0]['arguments']['session_id'], calls[1]['arguments']['session_id'])
        self.assertIn('Verified event', self.prompts[0][1]['content'])
        self.assertIn('https://example.org/news', self.prompts[0][1]['content'])
        self.assertIn('Page excerpts', self.prompts[0][1]['content'])
        methods = set()
        for request in self.fixture.requests:
            self.assertEqual(str(request.url), 'https://search.parallel.ai/mcp')
            self.assertEqual(request.headers['user-agent'], parallel.USER_AGENT)
            self.assertNotIn('authorization', request.headers)
            methods.add(json.loads(request.content)['method'])
        self.assertTrue({'initialize', 'tools/list', 'tools/call'} <= methods)

    def run_main(self, inputs, raw=False):
        with patch('builtins.input', side_effect=inputs), \
             patch.object(agent, 'detect_model', return_value=('test', 'scripted')), \
             patch.object(agent, 'get_current_model', return_value='test'), \
             patch.object(agent, 'print_banner'), \
             patch.object(agent, 'classify_intent', return_value='search') as classify, \
             patch.object(agent, 'stream_llm', return_value=iter(['plain response'])) as stream, \
             patch.object(agent, 'console', Console(file=io.StringIO(), width=100)):
            agent.main()
            if raw:
                classify.assert_not_called()
                stream.assert_called_once()
            return stream, agent.console.file.getvalue()

    def test_main_followup_clear_and_independent_conversations(self):
        self.run_main(['current event', 'follow up', '/clear', 'current event', '/quit'])
        ids = [x['arguments']['session_id'] for x in self.fixture.calls()]
        self.assertEqual(ids[:4], [ids[0]] * 4)
        self.assertEqual(ids[4:], [ids[4]] * 2)
        self.assertNotEqual(ids[0], ids[4])
        self.run_main(['another conversation', '/quit'])
        self.assertNotEqual(ids[4], self.fixture.calls()[-1]['arguments']['session_id'])

    def test_interleaved_conversations_and_followups(self):
        a, b = parallel.SearchConversation(), parallel.SearchConversation()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(agent.quick_search, query, conv)
                       for query, conv in [('alpha', a), ('beta', b), ('alpha again', a), ('beta again', b)]]
            for result in futures:
                self.assertIn('October 20', result.result()[0])
        search = [x['arguments'] for x in self.fixture.calls() if x['name'] == 'web_search']
        for args in search:
            self.assertEqual(args['session_id'], a.session_id if args['objective'].startswith('alpha') else b.session_id)
        self.assertNotEqual(a.session_id, b.session_id)

    def test_default_and_explicit_duckduckgo_do_not_connect_to_parallel(self):
        import ddgs
        for provider in (None, 'duckduckgo'):
            if provider is None:
                os.environ.pop('MAC_CODE_SEARCH_PROVIDER', None)
            else:
                os.environ['MAC_CODE_SEARCH_PROVIDER'] = provider
            with patch.object(ddgs, 'DDGS') as ddg:
                ddg.return_value.text.return_value = [{'title': 'Sports', 'body': '$123 and $456', 'href': 'https://example.org'}]
                ddg.return_value.news.return_value = []
                self.assertIsNotNone(agent.quick_search('prices'))
                ddg.assert_called_once()
        self.assertEqual(self.fixture.requests, [])

    def test_disabled_invalid_and_raw_do_not_use_tools(self):
        for value in ('none', 'invalid'):
            os.environ['MAC_CODE_SEARCH_PROVIDER'] = value
            with self.assertRaises(parallel.SearchError):
                agent.quick_search('event')
        os.environ['MAC_CODE_SEARCH_PROVIDER'] = 'parallel'
        self.run_main(['/raw', 'event', '/quit'], raw=True)
        self.assertEqual(self.fixture.requests, [])
        self.assertEqual(self.prompts, [])

    def test_provider_errors_do_not_generate_unsourced_answers_or_switch(self):
        with patch.object(parallel, 'REQUEST_TIMEOUT', 0.2):
            for failure in ('tool', 'http', 'empty', 'unsafe-url', 'malformed'):
                self.fixture.failure = failure
                with self.assertRaises(parallel.SearchError):
                    agent.quick_search('event')
            self.assertEqual(self.prompts, [])
            stream, output = self.run_main(['event', '/quit'])
            stream.assert_not_called()
            self.assertIn('timed out', output)

    def test_timeout_and_cancellation_close_inflight_request(self):
        self.fixture.failure = 'wait'
        with patch.object(parallel, 'REQUEST_TIMEOUT', 0.1):
            with self.assertRaisesRegex(parallel.SearchError, 'timed out'):
                agent.quick_search('event')
        self.assertTrue(self.fixture.closed)
        self.fixture.closed = False
        cancel = threading.Event()
        def stop():
            self.fixture.started.wait(2)
            cancel.set()
        self.fixture.started.clear()
        thread = threading.Thread(target=stop)
        thread.start()
        with self.assertRaisesRegex(parallel.SearchError, 'cancelled'):
            agent.quick_search('event', cancel_event=cancel)
        thread.join()
        self.assertTrue(self.fixture.closed)
        self.assertEqual(self.prompts, [])

    def test_main_interrupt_cancels_mcp_and_returns_to_prompt(self):
        self.fixture.failure = 'wait'
        real_sleep = agent.time.sleep
        def interrupt_search(delay):
            if self.fixture.started.is_set():
                raise KeyboardInterrupt
            real_sleep(delay)
        with patch.object(agent.time, 'sleep', side_effect=interrupt_search):
            stream, output = self.run_main(['event', '/quit'])
        stream.assert_not_called()
        self.assertIn('search cancelled', output)
        self.assertTrue(self.fixture.closed)
        self.assertEqual(self.prompts, [])

    def test_missing_optional_dependency_has_actionable_error(self):
        real_import = __import__
        def import_without_mcp(name, *args, **kwargs):
            if name == 'mcp':
                raise ImportError('optional package absent')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=import_without_mcp):
            with self.assertRaisesRegex(parallel.SearchError, 'requirements-search.txt'):
                agent.quick_search('event')
        self.assertEqual(self.fixture.requests, [])

    def test_retry_keeps_conversation_id(self):
        conv = parallel.SearchConversation()
        self.fixture.failure = 'tool'
        with self.assertRaises(parallel.SearchError):
            agent.quick_search('event', conv)
        self.fixture.failure = None
        agent.quick_search('event', conv)
        self.assertEqual({x['arguments']['session_id'] for x in self.fixture.calls()}, {conv.session_id})


if __name__ == '__main__':
    unittest.main()
