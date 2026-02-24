# Modified from the original implementation: https://github.com/NVlabs/vla0
# Authors: Gokul Puthumanaillam, Navya Gupta


import base64
import io
import json
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from rv_train.deploy.data_models import So100Base64DataModel
from rv_train.deploy.model_manager import So100ModelManager
from rv_train.model_specs import action_horizon as get_action_horizon_from_cfg

rbv_mm = None

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def rgb_from_base64(base64_string: str) -> np.ndarray:
    img = base64.b64decode(base64_string)
    array_bytes = io.BytesIO(img)
    return np.load(array_bytes)


def get_action_horizon(cfg) -> int:
    return get_action_horizon_from_cfg(cfg)


@app.post("/predict_base64")
async def predict(data: So100Base64DataModel):
    image_data = np.array([rgb_from_base64(d_rgb) for d_rgb in data.base64_rgb])
    state_data = np.array(data.state)
    instr_data = data.instr
    start_time = time.time()
    with torch.no_grad():
        output, _ = rbv_mm.forward(image_data, state_data, instr_data)
    print(f"Time taken: {time.time() - start_time}")
    return output


@app.post("/predict_base64_stream")
async def predict_base64_stream(data: So100Base64DataModel):
    image_data = np.array([rgb_from_base64(d_rgb) for d_rgb in data.base64_rgb])
    state_data = np.array(data.state)
    instr_data = data.instr

    def generate():
        start_time = time.time()
        last_action_txt = ""
        action_horizon = get_action_horizon(rbv_mm.cfg)
        for i in range(action_horizon):
            with torch.no_grad():
                output, last_action_txt = rbv_mm.forward(
                    image_data,
                    state_data,
                    instr_data,
                    get_one_step_action=True,
                    last_action_txt=last_action_txt,
                )
                print(last_action_txt)
            print(f"Time taken: {time.time() - start_time}")
            yield json.dumps({"index": i, "value": output}) + "\n"
        yield json.dumps({"time_taken": time.time() - start_time}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def get_ip_address():
    import socket

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


if __name__ == "__main__":
    rbv_mm = So100ModelManager()
    PORT = 10000
    print()
    print(f"IP address: {get_ip_address()}")
    print(f"Go to http://{get_ip_address()}:{PORT}/docs for the API documentation")
    print()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
