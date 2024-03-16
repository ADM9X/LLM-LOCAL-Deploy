# Practice Makes Perfect!!!
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).half().cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()


def predict(text):
    response, history = model.chat(tokenizer, text, history=[])
    return response


# chatbot_interface = gr.Chatbot()

# 创建Gradio接口
iface = gr.Interface(
    fn=predict,  # 指定要演示的函数
    inputs="text",  # 指定输入类型为文本
    outputs="text",  # 指定输出类型为文本
    description="输入问题，与机器人开始聊天吧！",
    api_name="/hypers.aigc.com/",

)

# 启动Gradio接口
iface.launch(debug=True, server_name="0.0.0.0", server_port=7860)
