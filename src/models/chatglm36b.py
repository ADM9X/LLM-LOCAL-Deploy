from .base import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ChatGLM(BaseModel):
    def __init__(self):
        super().__init__()
        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model_data = self.data_dir / "chatglm3-6b"  # 假设已经提前将模型数据下载到本目录
        self._load()

    def _load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_data, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_data,
            trust_remote_code=True,
            quantization_config=self.nf4_config,
            device_map='auto'
        ).cuda().eval()

    def _predict(self, input):
        for response, _ in self.model.stream_chat(self.tokenizer, input, history=[]):
            yield response

    def get_predict(self):
        return self._predict
