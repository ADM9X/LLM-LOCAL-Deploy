import argparse

import gradio as gr

from models.chatglm36b import ChatGLM
from models.baichuan import Baichuan


SUPPORTED_MODELS = {
    "chatglm3-6b": ChatGLM,
    "baichuan2-13b": Baichuan
}


def check_args(args):
    models_names = "\n  ".join(SUPPORTED_MODELS.keys())
    assert args.model in SUPPORTED_MODELS, \
        f"Unrecognized model parameter: '{args.model}'\n{'-'*80}\n" \
        f"Currently, only the following models are supported:\n  " \
        f'{models_names}'


def generate(args):
    check_args(args)
    model = SUPPORTED_MODELS[args.model]()
    predict = model.get_predict()

    gr.ChatInterface(predict).queue().launch(server_name="0.0.0.0")  # port 7860


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypers LLMs Demo")
    parser.add_argument('--model', required=True, type=str, help='Model Name')
    args = parser.parse_args()
    generate(args)
