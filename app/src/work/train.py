import torchvision.transforms.v2 as tfs 
from tqdm import tqdm 

from app.src.dataset import SunDataset

import torch.utils.data as data 

import torch 
from app.src.models import SunModel
from app.src.train import SunTrain

import torch.nn as nn 
import torch.optim as optim 

from app.src.config import Config



def train():
    # Создание dataset 
    transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale = True)])
    d_train = SunDataset(Config.PATH_DATASET, transform = transforms)
    train_data = data.DataLoader(d_train, batch_size = 32, shuffle = True)


    # Создание модели 
    model = SunModel().get_model()

    optimizer = optim.Adam(params = model.parameters(), lr = 0.001, weight_decay = 0.001)
    loss_function = nn.MSELoss()
    
    # Обучение модели
    train = SunTrain(
        train_data = train_data,
        epochs = 5,
        model = model,
        optimizer = optimizer,
        loss_function = loss_function
    )

    model = train.forward()

    # Сохранение весов модели 
    import os
    save_dir = os.path.dirname(Config.PATH_SAVE_MODEL)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    st = model.state_dict()
    torch.save(st, Config.PATH_SAVE_MODEL)


    # Тестируем нейросеть 
    d_test = SunDataset(Config.PATH_DATASET, train = False, transform = transforms)
    test_data = data.DataLoader(d_test, batch_size = 50, shuffle = False)

    Q = 0
    count = 0 
    model.eval()


    test_tqdm = tqdm(test_data, leave = True)
    for x_test, y_test in test_tqdm:
        with torch.no_grad():
            p = model(x_test)
            Q += loss_function(p, y_test).item()
            count += 1

    Q /= count 
    print(Q) 