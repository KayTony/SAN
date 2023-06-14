from typing_extensions import Annotated
import base64

import numpy as np
import cv2 as cv
import torch
# from torchvision.transforms import ToTensor
# from PIL import Image

from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words
from inference import convert

from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()
device = "cpu"

params = load_config("config.yaml")
words = Words(params['word_path'])
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words


def init_model():
    model = Backbone(params)
    model = model.to(device)
    load_checkpoint(model, None, params['checkpoint'])
    model.eval()
    return model


class Buffer(BaseModel):
    payload: str


@app.post("/api/san/prediction/")
async def predict(buffer: Buffer, model: Annotated[Backbone, Depends(init_model)]):
    img_buffer = base64.b64decode(buffer.payload)
    img_array = np.frombuffer(img_buffer, np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_ANYDEPTH)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = torch.Tensor(img) / 255
    image = image.unsqueeze(0).unsqueeze(0)

    image_mask = torch.ones(image.shape)
    image, image_mask = image.to(device), image_mask.to(device)
    prediction = model(image, image_mask)

    latex_list = convert(1, prediction)
    pred_latex = ' '.join(latex_list)
    return pred_latex