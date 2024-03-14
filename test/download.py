# pip install transformers torch sentencepiece
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 执行此脚本会自动从huggingface上下载模型数据到指定文件，本地部署时传入模型数据存放的路径即可
# 以"THUDM/chatglm3-6b"为例，执行程序时会自动从官网下载，并缓存到指定目录，此名称可以是任一官网模型名。
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, cache_dir="./data/chatglm3-6b/")  # cache_dir是指本地缓存目录，对应本项目则是./data/chatglm3-6b/
model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir="./data/chatglm3-6b/").cuda()

model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
