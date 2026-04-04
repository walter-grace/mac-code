"""
Module engine

This module provides lightweight adapters that present an OpenAI-compatible
interface for local llama.cpp. It contains two primary classes:

- LlamaCPPEngine: A front-facing adapter that accepts a JobInput and transforms
  it into OpenAI-compatible routes and payloads. It normalizes prompt vs chat
  input and delegates actual interaction to an OpenAI-style engine.
- LlamaCPPOpenAIEngine: A concrete engine that uses the OpenAI client (pointing
  at a locally hosted inference service) to list models and create completions
  or chat completions. It supports both non-streaming and streaming responses.

Environment:
- The OpenAI client is configured to point at a local base_url in this script.

Typical usage:
- Construct LlamaCPPEngine() and call its async generate() with a JobInput.
- The generator yields either dicts (non-stream responses) or streaming chunks
  formatted as strings when stream=True.
"""

import json

from dotenv import load_dotenv
from openai import OpenAI
from utils import JobInput

client = OpenAI(
    base_url="http://localhost:3098/v1/",
    api_key="",
)


class LlamaCPPEngine:
    """
    Adapter that prepares JobInput for an OpenAI-style engine and yields
    responses.

    This class normalizes incoming JobInput.llm_input into an OpenAI-style
    payload. If llm_input is a string, it prepares a completions request; if
    it's a list (messages), it prepares a chat.completions request. It then
    delegates the actual generation to a lower-level OpenAI-compatible engine
    (LlamaCPPOpenAIEngine).

    Attributes:
        None beyond temporary per-call variables — the class is stateless with
        respect to client configuration.
    """

    def __init__(self):
        """
        Initialize the LlamaCPPEngine.

        Loads environment variables (via python-dotenv). This constructor does
        not perform network calls and is intentionally lightweight to allow
        quick instantiation in worker processes.

        Example:
            engine = LlamaCPPEngine()
        """

        load_dotenv()
        print("Llama.cpp engine initialized")

    async def generate(self, job_input):
        """
        Asynchronously generate responses for a given JobInput.

        The method inspects job_input.llm_input and builds an OpenAI-compatible
        JobInput object that routes to either "/v1/completions" or
        "/v1/chat/completions". It then delegates to LlamaCPPOpenAIEngine to
        perform the actual request and yields each response chunk produced by
        that engine.

        Args:
            job_input (utils.JobInput): Input job object that contains at \
            least:
                - llm_input: either a prompt string (for completions) or a list
                  of chat messages (for chat completion).
                - stream: boolean indicating whether a streaming response is
                  desired.

        Yields:
            dict or str: Each yielded item is either:
                - a dict representing a non-streaming response,
                - a string that contains streaming "data: ..." chunks for
                  stream=True,
                - or an error dict with an "error" key.

        Notes:
            - This method does not validate the full schema of job_input; it
              relies on upstream code to supply a correctly shaped JobInput.
            - The default model selection uses the first model listed by the
              server upon requesting /v1/models.
        """

        openAIEngine = LlamaCPPOpenAIEngine()

        # Get model to use (defaults to first model in list of models)
        model = client.models.list().data[0].id

        # Depending if prompt is a string or a list, we need to handle it
        # differently and send it to the OpenAI API
        if isinstance(job_input.llm_input, str):
            # Build new JobInput object with the OpenAI route and input
            openAIjob = JobInput(
                {
                    "openai_route": "/v1/completions",
                    "openai_input": {
                        "model": model,
                        "prompt": job_input.llm_input,
                        "stream": job_input.stream,
                    },
                }
            )
        else:
            # Build new JobInput object with the OpenAI route and input
            openAIjob = JobInput(
                {
                    "openai_route": "/v1/chat/completions",
                    "openai_input": {
                        "model": model,
                        "messages": job_input.llm_input,
                        "stream": job_input.stream,
                    },
                }
            )

        print("Generating response for job_input:", job_input)
        print("OpenAI job:", openAIjob)

        # Create a generator that will yield the response from the OpenAI API
        generate = openAIEngine.generate(openAIjob)

        # Yield the response from the OpenAI API
        async for batch in generate:
            yield batch


class LlamaCPPOpenAIEngine(LlamaCPPEngine):
    """
    Concrete OpenAI-compatible engine that uses the OpenAI client to perform
    model listing and completions against a locally hosted inference service.

    This class expects the module-level `client` object to be configured to
    point to a compatible service (llama.cpp).
    that implements the same interface as the official OpenAI Python client.

    Behavior:
        - For route "/v1/models": lists available models and yields a dict
          describing them.
        - For "/v1/chat/completions" or "/v1/completions": performs a request
          and yields either a single response (non-stream) or streaming chunks.
    """

    def __init__(self):
        """
        Initialize the LlamaCPPOpenAIEngine.

        Loads environment variables and prints an initialization message. No
        network calls are made here.

        Example:
            engine = LlamaCPPOpenAIEngine()
        """

        load_dotenv()
        print("LlamaCPPOpenAIEngine initialized")

    async def generate(self, job_input):
        """
        Dispatch the provided OpenAI-style JobInput to the appropriate handler.

        Args:
            job_input (JobInput): JobInput with at least:
                - openai_route: str, one of "/v1/models",
                  "/v1/chat/completions", "/v1/completions"
                - openai_input: dict containing the parameters for the chosen
                  route.

        Yields:
            dict or str: Each yielded value can be:
                - A dictionary representing the full response (non-streaming).
                - A sequence of strings for streaming responses (prefixed with
                  "data: ").
                - An error dict if the route is invalid or an exception occurs.
        """

        # Dump job_input to console
        openai_input = job_input.openai_input

        # for now e just mock the response
        if job_input.openai_route == "/v1/models":
            # Async response
            async for response in self._handle_model_request():
                yield response
        elif job_input.openai_route in [
            "/v1/chat/completions",
            "/v1/completions",
        ]:
            async for response in self._handle_chat_or_completion_request(
                openai_input,
                chat=job_input.openai_route == "/v1/chat/completions",
            ):
                yield response
        else:
            yield {"error": "invalid route"}

    async def _handle_model_request(self):
        """
        Handle a model-listing request using the OpenAI client.

        This method calls client.models.list() and yields a single dict that
        mirrors the structure of OpenAI's model list responses:
            {"object": "list", "data": [<model dicts>]}

        Yields:
            dict: The model list as a dictionary, or an error dict on failure.

        Raises:
            No exceptions are propagated; errors are converted to
            `{"error": str(e)}`.
        """

        try:
            response = client.models.list()

            yield {
                "object": "list",
                "data": [model.to_dict() for model in response.data],
            }
        except Exception as e:
            yield {"error": str(e)}

    async def _handle_chat_or_completion_request(
        self, openai_input, chat=False
    ):
        """
        Handle a chat or completion request and yield responses or streaming
        chunks.

        This method chooses between client.chat.completions.create and
        client.completions.create based on the 'chat' flag. If the input
        requests non-streaming behavior (openai_input.get("stream") is falsy),
        it yields a single dict representation of the response. If streaming is
        requested, it iterates over the streaming response and yields
        JSON-formatted chunks prefixed with "data: ", matching common SSE-like
        streaming conventions.

        Args:
            openai_input (dict): Parameters to pass to the client's create
                method.
            chat (bool): If True, call the chat completion endpoint; otherwise,
                call the standard completion endpoint.

        Yields:
            dict or str:
                - For non-stream: yields the full response as a dict.
                - For stream: yields strings of the form "data: <json>\\n\\n"
                  for each chunk, and finally "data: [DONE]".
                - On error: yields {"error": "<message>"}.

        Notes:
            - Exceptions are caught and yielded as error dictionaries so the
              async generator consumer can handle them without dealing with
              exceptions.
            - The exact behavior depends on the local client implementation
              being OpenAI-compatible.
        """

        try:
            # Call openai.chat.completions.create or openai.completions.create
            # based on the route
            if chat:
                response = client.chat.completions.create(**openai_input)
            else:
                response = client.completions.create(**openai_input)

            # If streaming is False, we can just return the response
            if not openai_input.get("stream", False):
                yield response.to_dict()
                return

            for chunk in response:
                # Return json of the chunk without any line breaks
                yield "data: " + json.dumps(
                    chunk.to_dict(), separators=(",", ":")
                ) + "\n\n"

            yield "data: [DONE]"

        except Exception as e:
            yield {"error": str(e)}
