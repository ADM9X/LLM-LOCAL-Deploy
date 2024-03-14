from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse
# 初始化参数构造器
parser = argparse.ArgumentParser()
# 在参数构造器中添加两个命令行参数
parser.add_argument('--prompt', type=str,default="你好！")
# 获取所有的命令行参数
args = parser.parse_args()

prompt = args.prompt

path = './data/chatglm3/'  # 本地模型所在位置。设置为"THUDM/chatglm3-6b"时会自动下载。

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).half().cuda()  # half是量化模式，cuda是将模型放到GPU上

# 方式1
model = model.eval()
response, history = model.chat(tokenizer, prompt, history=[])
print(response)
