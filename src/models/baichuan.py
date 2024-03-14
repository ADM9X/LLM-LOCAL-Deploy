from .base import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


class Baichuan(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_data = self.data_dir / "Baichuan2-13B"
        self._load()

    def _load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_data,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).cuda().eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_data
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_data,
            use_fast=False,
            trust_remote_code=True
        )

    @staticmethod
    def _build_msgs(input, history):
        msgs = []
        for item in history:
            prompt, response = item
            msgs.append({"role": "user", "content": prompt})
            msgs.append({"role": "assistant", "content": response})
        msgs.append({"role": "user", "content": input})
        return msgs

    def _predict(self, input):
        msgs = self._build_msgs(input, history=[])
        for response in self.model.chat(self.tokenizer, msgs, stream=True):
            yield response
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def get_predict(self):
        return self._predict
