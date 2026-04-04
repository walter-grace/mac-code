"""
This module contains utility classes and functions for handling job inputs.
"""


class JobInput:
    """
    Class to parse and store job input data. It extracts fields such as
    llm_input, stream, openai_route, and openai_input from the provided job
    dictionary.
    """

    def __init__(self, job):
        """
        Initialize the JobInput instance by parsing the job dictionary.

        Default values:
        - llm_input: job["messages"] if present, else job["prompt"]
        - stream: False
        - openai_route: None
        - openai_input: None
        """

        self.llm_input = job.get("messages", job.get("prompt"))
        self.stream = job.get("stream", False)
        self.openai_route = job.get("openai_route")
        self.openai_input = job.get("openai_input")
