
from tqdm import tqdm 
from app.src.models import SunModel

class SunTrain:
    def __init__(self, train_data, epochs: int, model: SunModel, optimizer, loss_function):
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_data = train_data
    
    def forward(self) -> SunModel:
        self.model.train()

        for _e in range(self.epochs):
            loss_mean = 0
            lm_count = 0

            train_tqdm = tqdm(self.train_data, leave = True)

            for x_train, y_train in train_tqdm:
                predict = self.model(x_train)
                loss = self.loss_function(predict, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean 
                train_tqdm.set_description(f"Epoch [{_e + 1}/{self.epochs}], loss_mean = {loss_mean:.3f}")
        
        return self.model