import json
from pathlib import Path

from evaluations.llm_as_a_judge import LLMJudge
class Summarizer:
    def __init__(
        self,
        llm=LLMJudge
    ):
        self.llm = llm
        base_dir = Path(__file__).parent
        prompts_dir = base_dir / "prompts" 
        self._prompt_template = (prompts_dir/"summary_prompt.txt").read_text(encoding="utf-8")

    def generate_overall_summary(self, summary_dict):
        string_dict = {str(key): value for key, value in summary_dict.items()}
        eval_json = json.dumps(string_dict)
        prompt = self._prompt_template.format(eval_summary=eval_json)
        summary = self.llm.generate(prompt)
        return summary
        