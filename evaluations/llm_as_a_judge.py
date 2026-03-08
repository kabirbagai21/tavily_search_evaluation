from openai import OpenAI


class LLMJudge:
    """
    Simple OpenAI-backed LLM judge wrapper.

    Only requires a model name.
    Assumes OPENAI_API_KEY is already set in the environment.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
        )
        return response.output_text
