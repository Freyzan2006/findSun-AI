


from .work import test, train

def main():

    while True:
        user_input: int = int(input("1 - Обучить\n2 - Тестировать\n3 - Выход\n"))

        if user_input == 1: 
            train() 
        elif user_input == 2:
            test() 
        elif user_input == 3:
            print("Вы вышли из программы")
            break
        else:
            print("Такой команды нет")







