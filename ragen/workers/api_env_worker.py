import ray
from openai import OpenAI
import logging
from typing import List
import concurrent.futures

logger = logging.getLogger(__name__)


class APIEnvironmentLLMWorker:
    def __init__(self, config,role=None):
        self.config = config
        self.role=role
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=config.get("base_url"),
            api_key=config.get("api_key")
        )
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.6)
        self.max_tokens = config.get("max_tokens", 512)
        
        # 并发控制
        self.concurrency = config.get("concurrency", 16)

    def init_model(self):
        # API 模式不需要加载权重，打印一下日志即可
        print(f"[API Worker] Initialized with model: {self.model_name} at {self.client.base_url}")

    def _call_api(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API Call Error: {e}")
            return "Sorry, I cannot answer right now."

    def generate_responses(self, prompts: List[str]) -> List[str]:
        """
        接收字符串列表，返回字符串列表。
        使用线程池并发请求，提高 RL 训练时的吞吐量。
        """
        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {executor.submit(self._call_api, prompt): i for i, prompt in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    data = future.result()
                    results[idx] = data
                except Exception as exc:
                    logger.error(f"Generated an exception: {exc}")
                    results[idx] = "Error"
        
        return results