import gradio as gr
import argparse
from src.models.chatglm36b import ChatGLM
from src.models.baichuan import Baichuan

SUPPORTED_MODELS = {
    "chatglm3-6b": ChatGLM,
    "baichuan2-13b": Baichuan
}


def check_args(args):
    models_names = "\n  ".join(SUPPORTED_MODELS.keys())
    assert args.model in SUPPORTED_MODELS, \
        f"Unrecognized model parameter: '{args.model}'\n{'-' * 80}\n" \
        f"Currently, only the following models are supported:\n  " \
        f'{models_names}'


parser = argparse.ArgumentParser(description="Hypers LLMs Demo")
parser.add_argument('--model', required=True, type=str, help='Model Name', default="chatglm3-6b")
args = parser.parse_args()
check_args(args.model)
model = SUPPORTED_MODELS[args.model]
predict = model().get_predict()


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
