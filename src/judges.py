import os
import re
from openai import OpenAI
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from src.constants import USER_PROMPT_OPENAI, SCORING_RULES, DETAILED_POLICY, REFUSAL_PROMPT
from tqdm import tqdm


class Judge(ABC):

    def __init__(
        self,
        model: str,
        system_prompt: str | None,
        user_prompt: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        resolved_base_url = base_url or os.environ.get("JUDGE_BASE_URL", None)
        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)

    @abstractmethod
    def _parse_response(self, response: str):
        pass

    def judge(self, inputs: dict[str, str], temperature: float = 0.0):
        messages = [] if self.system_prompt is None else [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": self.user_prompt.format(**inputs)})
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                n=1,
                temperature=temperature,
                messages=messages
            )
            text = completion.choices[0].message.content
            return self._parse_response(text)
        except Exception as e:
            print(f"Error calling judge API ({self.model}): {e}")
            return None

    def judge_batch(
        self,
        inputs: list[dict[str, str]],
        temperature: float = 0.0,
        max_workers: int | None = 100
    ) -> list:
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

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        system_prompt = None
        user_prompt = USER_PROMPT_OPENAI
        super().__init__(model, system_prompt, user_prompt, api_key=api_key, base_url=base_url)

    def _parse_response(self, response: str) -> int | None:
        if not response:
            return None
        if "#thescore:" in response:
            rating = response.split("#thescore:")[1]
        else:
            rating = response
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        if not digit_groups:
            return None
        try:
            return int(float(digit_groups[0]))
        except Exception:
            return None

    def judge(self, inputs: dict[str, str], temperature: float = 0.0):
        inputs["scoring_rules"] = SCORING_RULES
        inputs["detailed_policy"] = DETAILED_POLICY
        return super().judge(inputs, temperature)


class RefusalJudge(Judge):

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        system_prompt = None
        user_prompt = REFUSAL_PROMPT
        super().__init__(model, system_prompt, user_prompt, api_key=api_key, base_url=base_url)

    def _parse_response(self, response: str) -> str:
        clean = response.split("</think>")[-1] if "</think>" in response else response
        matches = re.findall(r"\b([a-d])\b", clean.lower())
        return matches[-1] if matches else clean.strip().lower()

    def judge(self, inputs: dict[str, str], temperature: float = 0.0):
        return super().judge(inputs, temperature)