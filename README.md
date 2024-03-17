## LLM LOCAL Deployment2
 LLM LOCAL Deployment2 is a project designed to demonstrate the local deployment of the open-source LLM (Large Language Model). It provides a web interface to showcase the capabilities of the model.  
## Table of Contents
Description

Download Model From HuggingFace Hub

Installation

Useage

Models
## Description
 LLM LOCAL Deployment2 allows you to deploy the LLM model locally for quantitative tasks. With this project, you can easily set up and run the LLM model on your machine. It provides a user-friendly web interface to interact with the model and explore its capabilities.  
## Download Model From HuggingFace Hub
### Mode1：Use mirror to download model data to local
```bash
# vim ~/.bashrc
export HF_ENDPOINT = "https://hf-mirror.com"  # 设置下载镜像
export HF_HOME = ""  # 设置下载目录

# source ~/.bashrc

# 终端执行下载命令，其中THUDM/chatglm3-6b是指HuggingFace上模型的Name
# pip install -U huggingface_hub
huggingface-cli download --resume-download THUDM/chatglm3-6b --local-dir /home/llm/hugging/chatglm6b
```
### Mode2：Obtain automatically using transformers module
Downloading model data is very susceptible to network impacts, and after each request timeout, the download script can be executed again until the download is complete

```python
# pip install transformers torch sentencepiece
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 执行此脚本会自动从huggingface上下载模型数据到指定文件，本地部署时传入模型数据存放的路径即可
# 以"THUDM/chatglm3-6b"为例，执行程序时会自动从官网下载，并缓存到指定目录，此名称可以是任一官网模型名。
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True,
                                          cache_dir="data/chatglm3-6b/")  # cache_dir是指本地缓存目录，对应本项目则是./data/chatglm3-6b/
model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             cache_dir="data/chatglm3-6b/").cuda()

model = model.eval()

respone, history = model.chat(tokenizer, "你好", history=[])
print(respone)
```
（Note: You may need to refer to the content of this section to download the model to the local in advance.）
## Installation
### 1、Use Git command to download this project to local
```bash
git clone ...
```
### 2、Create a Python virtual environment
eg: Using the Anaconda environment
```bash
# your_env_name是待创建的虚拟环境名称，可随意修改
conda create -n your_env_name python=3.10
conda activate your_env_name
```
### 3、Install dependencies
```bash
pip install -r requirements.txt
```
## Useage
### Mode1：Run python web_run.py directly in the virtual environment to chat with LLM
The prerequisite is that the model data has been downloaded from the data directory, Otherwise, you can try using web_download_derictly.py files
```bash
python web_load_local.py
```
### Mode2: Run python api.py directly in the virtual environment to run unvicorn server
The prerequisite is that the model data has been downloaded from the data directory, Otherwise, you can try using api_down_derictly.py files
```bash
uvicorn api_load_local:app --reload --host 0.0.0.0 --port 8002
```
## Models
The LLM Local Deploy project supports the following models:

- **chatglm3_6b**: Model description.
- **baichuan2_13b**: Model description.

Feel free to experiment with different models and their capabilities.
## Reference
* Using transformers: https://huggingface.co/docs/transformers/pipeline_tutorial
* Using gradio: https://www.gradio.app/
* Anaconda installation on linux: https://zhuanlan.zhihu.com/p/440548295

