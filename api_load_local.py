import datetime
import json
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SUPPORTED_MODELS = [
    "chatglm3-6b",
    "baichuan2-13b"
]


def check_args(model):
    models_names = "\n  ".join(SUPPORTED_MODELS)
    assert model in SUPPORTED_MODELS, \
        f"Unrecognized model parameter: '{model}'\n{'-' * 80}\n" \
        f"Currently, only the following models are supported:\n  " \
        f'{models_names}'


app = FastAPI()


@app.post("/hypers.aigc.com/")
async def generate(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    prompt = json_post_list.get('prompt')
    model = json_post_list.get('model')
    check_args(model)

    path = f"./data/{model}/"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).half().cuda()
    model = model.eval()

    response, history = model.chat(tokenizer, prompt, history=[])
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer
