import os
import re
from openai import OpenAI
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from src.constants import USER_PROMPT_OPENAI, SCORING_RULES, DETAILED_POLICY, REFUSAL_PROMPT
from tqdm import tqdm


class Judge(ABC):

    def __init__(self, model: str, system_prompt: str | None, user_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @abstractmethod
    def _parse_response(self, response: str) -> int:
        pass

    def judge(self, inputs: dict[str, str], temperature: float = 0.0) -> int:
        messages = [] if self.system_prompt is None else [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": self.user_prompt.format(**inputs)})
        completion = self.client.chat.completions.create(
            model=self.model,
            n=1,
            temperature=temperature,
            messages=messages
        )
        text = completion.choices[0].message.content
        return self._parse_response(text)

    def judge_batch(
        self,
        inputs: list[dict[str, str]],
        temperature: float = 0.0,
        max_workers: int | None = 100
    ) -> list[int]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(lambda x: self.judge(x, temperature), inputs),
                    total=len(inputs),
                    desc="Judging batch"
                )
            )
        return results

        

class JailbreakJudge(Judge):

    def __init__(self, model: str):
        system_prompt = None
        user_prompt = USER_PROMPT_OPENAI
        super().__init__(model, system_prompt, user_prompt)

    def _parse_response(self, response: str) -> int:
        if "#thescore:" in response:
            rating = response.split("#thescore:")[1]
        else:
            rating = response
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return int(digit_groups[0])

    def judge(self, inputs: dict[str, str], temperature: float = 0.0) -> int:
        inputs["scoring_rules"] = SCORING_RULES
        inputs["detailed_policy"] = DETAILED_POLICY
        return super().judge(inputs, temperature)
    
class RefusalJudge(Judge):

    def __init__(self, model: str):
        system_prompt = None
        user_prompt = REFUSAL_PROMPT
        super().__init__(model, system_prompt, user_prompt)

    def _parse_response(self, response: str) -> int:    
        return response.lower()
    
    def judge(self, inputs: dict[str, str], temperature: float = 0.0) -> int:
        return super().judge(inputs, temperature)
    