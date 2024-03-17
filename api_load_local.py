import datetime
import json
from fastapi import FastAPI, Request
import argparse
SUPPORTED_MODELS = [
    "chatglm3-6b",
    "baichuan2-13b"
]


def check_args(args):
    models_names = "\n  ".join(SUPPORTED_MODELS)
    assert args.model in SUPPORTED_MODELS, \
        f"Unrecognized model parameter: '{args.model}'\n{'-' * 80}\n" \
        f"Currently, only the following models are supported:\n  " \
        f'{models_names}'


parser = argparse.ArgumentParser(description="Hypers LLMs Demo")
parser.add_argument('--model', required=True, type=str, help='Model Name', default="chatglm3-6b")
args = parser.parse_args()
check_args(args)
llm = args.model

path = f"./data/{llm}/"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).half().cuda()

app = FastAPI()


@app.post("/hypers.aigc.com/")
async def generate(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    prompt = json_post_list.get('prompt')

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
