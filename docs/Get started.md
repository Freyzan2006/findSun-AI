Начать (Желательно следовать сверху вниз):

Установка:
1. python -m venv .venv 
2. .venv\Scripts\Activate.bat (если на window)
3. pip install -r requirements.txt
4. скачайте pyTorch под вашу OC (https://pytorch.org/get-started/locally/)

Установка dataset:
1. pip install pygame
2. cd app/data/data_gen/
3. python dataset_reg_gen.py
4. pip uninstall pygame 
5. cd ../../../

Запуск:
    - python run.py 
    - docker-compose up --build 
