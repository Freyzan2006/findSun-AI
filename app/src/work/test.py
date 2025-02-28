from PIL import Image
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs

from app.src.models import SunModel
from app.src.config import Config 

def test():

    # Создание модели 
    model = SunModel().get_model()

    path = './app/data/data_set/test/'
    num_img = 100

    st = torch.load(Config.PATH_SAVE_MODEL, weights_only=False)
    model.load_state_dict(st)

    with open(os.path.join(path, "format.json"), "r") as fp:
        format = json.load(fp)

    transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
    img = Image.open(os.path.join(path, f'sun_reg_{num_img}.png')).convert('RGB')
    img_t = transforms(img).unsqueeze(0)

    model.eval()
    predict = model(img_t)
    print(predict)
    print(tuple(format.values())[num_img-1])
    p = predict.detach().squeeze().numpy()

    plt.imshow(img)
    plt.scatter(p[0], p[1], s=20, c='r')
    plt.show() 