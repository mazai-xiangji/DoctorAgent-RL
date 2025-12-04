import ray
from openai import OpenAI
import logging
from typing import List, Union, Dict
import concurrent.futures
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl import DataProto
import numpy as np

logger = logging.getLogger(__name__)

class APIEnvironmentLLMWorker(Worker):
    def __init__(self, config, role=None):
        super().__init__()
        self.config = config
        self.role = role
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=config.get("base_url"),
            api_key=config.get("api_key")
        )
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.6)
        self.max_tokens = config.get("max_tokens", 512)
        
        # Concurrency control
        self.concurrency = config.get("concurrency", 16)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # API mode doesn't need to load weights, just log
        print(f"[API Worker] Initialized with model: {self.model_name} at {self.client.base_url}")

    def _call_api(self, prompt: str) -> str:
        try:
            # Handle prompt format
            if isinstance(prompt, list):
                messages = prompt
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API Call Error: {e}")
            return "Sorry, I cannot answer right now."

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """
        Receives DataProto, extracts prompts from non_tensor_batch['prompts'],
        calls API concurrently, and returns DataProto with responses in non_tensor_batch['responses'].
        """
        # Extract prompts
        # Assuming prompts are passed in non_tensor_batch['prompts'] as a list or numpy array
        # prompts = data.non_tensor_batch.get('prompts')
        # if prompts is None:
        #     # Fallback check if passed differently (though env should pack it)
        #     logger.warning("No prompts found in non_tensor_batch['prompts']")
        #     return DataProto.from_dict(non_tensor_batch={'responses': []})

        # if isinstance(prompts, np.ndarray):
        #     prompts = prompts.tolist()

        results = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {executor.submit(self._call_api, prompt): i for i, prompt in enumerate(prompts)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    data_res = future.result()
                    results[idx] = data_res
                except Exception as exc:
                    logger.error(f"Generated an exception: {exc}")
                    results[idx] = "Error"
        
        # Return as DataProto
        return results