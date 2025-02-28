Начать (Желательно следовать сверху вниз):

Установка:
1. python -m venv .venv 
2. .venv\Scripts\Activate.bat (если на window)
3. pip install -r requirements.txt

Установка dataset:
1. pip install pygame
2. python app/data/data_gen/dataset_reg_gen.py
3. pip uninstall pygame 

Запуск:
    - python run.py 
    - docker-compose up --build 
