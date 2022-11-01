import random
import numpy as np

        
class neuron:

    def __init__(self, name, number, layer, value, width_next_layer):
        self.name = name
        self.number = number
        self.layer = layer
        self.value = value
        self.error = 0
        self.data = []
        self.error_data = []
        self.W = []
        self.dW = []
        for i in range(width_next_layer):
            self.W.append(random.random())
        for i in range(width_next_layer):
            self.data.append(0)
        for i in range(width_next_layer):
            self.dW.append(0)
        for i in range(width_next_layer):
            self.error_data.append(0)

    def logistic_function(self, x):
        return .5 * (1 + np.tanh(.5 * x))

    def generate_error(self, next_layer_error):
        self.error = 0
        for i in range(len(self.W)):
            self.error += next_layer_error[i] * self.W[i]

    def generate_value(self, data_array):
        data_sum = 0
        for i in range(len(data_array)):
            data_sum += data_array[i]
        self.value = self.logistic_function(data_sum)
   
    def weight_deflection(self, next_layer_value, next_layer_error):
        for i in range(len(self.dW)):
            self.dW[i] = next_layer_error[i]*(next_layer_value[i] * (1 - next_layer_value[i])) * self.value
        for i in range(len(self.W)):
            self.W[i] = self.W[i] + self.dW[i]

    def generate_data(self):
        for i in range(len(self.W)):
            self.data[i] = self.value * self.W[i]       

    def display_info(self):
        print("------------| Info about", self.name, "|------------",
              "\nName: ", self.name, 
              "\nNumber: ", self.number, "\nLayer: ", self.layer,
              "\nSelf value: ", self.value, "\nSelf error: ", self.error,
              "\nSelf data: ", self.data, "\nWeights: ", self.W,
              "\nLast delta weights: ", self.dW,
              "\n------------| Info about", self.name, "|------------")
        
    def set_initial_settings(self, width_next_layer):
        for i in range(width_next_layer):
            self.W.append(random.random())
        for i in range(width_next_layer):
            self.dW.append(0)


class neunet:
    
    def __init__(self, name, layers):
        self.name = name
        self.layers = []
        self.matrix = []
        self.bias = []
        for i in range(len(layers)):
            try:
                if layers[i] < 1: 
                    print("Обнаружена ошибка при вводе количества нейронов в слое.",
                          "Слой", i, "со значением", layers[i], "удален,"
                          "инициалзация нейронной сети будет проходить без него.",
                          "Будьте внимательны, значения слоев должны быть больше 0.")
                else:
                    self.layers.append(int(layers[i]))
            except:
                print("Обнаружена ошибка при вводе количества нейронов в слое.",
                      "Введенные данные неинтерпретируемы в целочисленный тип данных.",
                      "Слой", i, "со значением", layers[i], "удален,",
                      "инициалзация нейронной сети будет проходить без него.")
                
        print("Начинаю генерировать матрицу архитектуры сети ", self.layers)
        
        name = "neuron_l-"
        for i in range(len(self.layers)):
            self.matrix.append([])
        for i in range(len(self.layers)):
            layername = name + str(i) + "_n-"
            try:
                width_next_layer = self.layers[i + 1]
            except:
                width_next_layer = 0
            for j in range(self.layers[i]):
                self.matrix[i].append(neuron(name = layername + str(j),
                            number = j, layer = i,
                            value = 0, width_next_layer = width_next_layer))

        print("Генерирую массив нейронов смещения")
        name = "bias_l-"
        for i in range(len(self.layers) - 1):
            try:
                width_next_layer = self.layers[i + 1]
            except:
                width_next_layer = 0
            self.bias.append(neuron(name = name + str(i),
                            number = -1, layer = i,
                            value = 1, width_next_layer = width_next_layer))
        
        print("Сеть необходимой архитектуры инициализирована")

    def learn_round(self, input_data, output_data):
        
        if len(input_data) == len(self.matrix[0]):
            if len(output_data) == len(self.matrix[-1]):
                
                for j in range(len(input_data)):
                    self.matrix[0][j].value = input_data[j]
                for i in range(len(self.matrix)):
                    if i < len(self.matrix) - 1:
                        self.bias[i].generate_data()
                    for j in range(len(self.matrix[i])):
                        if i > 0:
                            data_array = []
                            for k in range(len(self.matrix[i-1])):
                                data_array.append(self.matrix[i-1][k].data[j])
                            data_array.append(self.bias[i-1].data[j])
                            self.matrix[i][j].generate_value(data_array)
                        if i < len(self.matrix):
                            self.matrix[i][j].generate_data()

                for j in range(len(output_data)):
                    self.matrix[-1][j].error = output_data[j] - self.matrix[-1][j].value
                for i in range(len(self.matrix)):
                    if i > 0:
                        bias_error_array = []
                        for k in range(len(self.matrix[-i])):
                            bias_error_array.append(self.matrix[-i][k].error)
                        self.bias[-i].generate_error(next_layer_error = bias_error_array)
                        
                    for j in range(len(self.matrix[-1-i])):
                        if i > 0:
                            error_array = []
                            for k in range(len(self.matrix[-i])):
                                error_array.append(self.matrix[-i][k].error)
                            self.matrix[-1-i][j].generate_error(next_layer_error = error_array)
                
                for i in range(len(self.matrix) - 1):
                    next_layer_bias_value = []
                    next_layer_bias_error = []
                    for k in range(len(self.matrix[i+1])):
                        next_layer_bias_value.append(self.matrix[i+1][k].value)
                        next_layer_bias_error.append(self.matrix[i+1][k].error)
                    self.bias[i].weight_deflection(next_layer_bias_value, next_layer_bias_error)
                    for j in range(len(self.matrix[i])):
                        next_layer_value = []
                        next_layer_error = []
                        for k in range(len(self.matrix[i+1])):
                            next_layer_value.append(self.matrix[i+1][k].value)
                            next_layer_error.append(self.matrix[i+1][k].error)
                        self.matrix[i][j].weight_deflection(next_layer_value, next_layer_error)

            elif len(output_data) > len(self.matrix[-1]):
                print("Введено слишком много выходных данных.",
                  "\nДлина массива выходных данных:", len(output_data),
                  "\nНейронов", len(self.matrix[-1]), "- го слоя:", len(self.matrix[-1]),
                  "\nЦикл обучения отменен.")
            else:
                print("Введено недостаточно выходных данных.",
                  "\nДлина массива выходных данных:", len(output_data),
                  "\nНейронов", len(self.matrix[-1]), "- го слоя:", len(self.matrix[-1]),
                  "\nЦикл обучения отменен.")
                
        elif len(input_data) > len(self.matrix[0]):
            print("Введено слишком много входных данных.",
                  "\nДлина массива входных данных:", len(input_data),
                  "\nНейронов нулевого слоя:", len(self.matrix[0]),
                  "\nЦикл обучения отменен.")
        else:
            print("Введено недостаточно входных данных.",
                  "\nДлина массива входных данных:", len(input_data),
                  "\nНейронов нулевого слоя:", len(self.matrix[0]),
                  "\nЦикл обучения отменен.")
            
    def predict_round(self, input_data):
        
        if len(input_data) == len(self.matrix[0]):
            
            for j in range(len(input_data)):
                self.matrix[0][j].value = input_data[j]
            for i in range(len(self.matrix)):
                if i < len(self.matrix) - 1:
                    self.bias[i].generate_data()
                for j in range(len(self.matrix[i])):
                    if i > 0:
                        data_array = []
                        for k in range(len(self.matrix[i-1])):
                            data_array.append(self.matrix[i-1][k].data[j])
                        data_array.append(self.bias[i-1].data[j])
                        self.matrix[i][j].generate_value(data_array)
                    if i < len(self.matrix):
                        self.matrix[i][j].generate_data()
                            
            prediction_data = []
            for j in range(len(self.matrix[-1])):
                prediction_data.append(self.matrix[-1][j].value)
            return prediction_data

        elif len(input_data) > len(self.matrix[0]):
            print("Введено слишком много входных данных.",
                  "\nДлина массива входных данных:", len(input_data),
                  "\nНейронов нулевого слоя:", len(self.matrix[0]),
                  "\nЦикл предсказания отменен.")
        else:
            print("Введено недостаточно входных данных.",
                  "\nДлина массива входных данных:", len(input_data),
                  "\nНейронов нулевого слоя:", len(self.matrix[0]),
                  "\nЦикл предсказания отменен.")

    def display_info(self):
        for i in range(len(self.matrix)):
            if i < len(self.matrix) - 1:
                self.bias[i].display_info()
            for j in range(len(self.matrix[i])):
                self.matrix[i][j].display_info()
