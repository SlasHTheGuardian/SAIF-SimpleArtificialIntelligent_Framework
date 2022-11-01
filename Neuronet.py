import SAIF as saif
import random


layers = [2, 10, 10, 1]  # Архитектура нейросети по слоям
neunet1 = saif.neunet("neural network 1 test", layers)

answer = 0
i = 0
num_of_learning_cycles = 1000000

percent = num_of_learning_cycles/100
percentage_step = percent

print("Начинаю обучение")
while i < num_of_learning_cycles:
    x1 = random.randint(-1000, 1000)/100
    y1 = random.randint(-1000, 1000)/100
    y = x1**2
    if y1 >= y:
        answer = 1
    else:
        answer = 0
    neunet1.learn_round([x1, y1], [answer])
    i += 1
print("Обучение завершено! Начнем работу.")
k = 1
while k != 0:
    print("Введите Х")
    x = float(input())
    print("Введите Y")
    y = float(input())
    print("Предсказанный результат:", neunet1.predict_round([x, y]), "\nПродолжаем?")
    k = int(input())
    if k == 0:
        print("Завершение работы")
    else:
        print("Окей")
