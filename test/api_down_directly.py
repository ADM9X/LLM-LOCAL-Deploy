import datetime
import json
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, truncation=True)
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).half().cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()


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
