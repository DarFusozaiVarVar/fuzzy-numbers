from StatAnalysis import *
from math import *
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import copy
with open("Москва_2021.txt", "r") as f:
    d_row = create_discrete_series([int(l.strip()) for l in f])

t_row = [[4922.4, 23369, 2404.8],
         [4130.7, 26629, 2302.2],
         [4137.4, 29792, 2206.2],
         [3889.4, 32495, 2190.6],
         [4263.9, 34030, 2388.5],
         [4243.5, 36709, 2160.1],
         [3966.5, 39167, 2058.5],
         [3657, 43724, 1991.5],
         [3461.2, 47867, 2024.3],
         [4316, 51344, 2044.2],
         [3624.6, 57244, 2004.4]]
    
##def MNK(row, m = 1):
##    C = np.array([[1.0] + c[:m] for c in row])
##    y = np.array([c[1] for c in row])
##    O_array = np.linalg.inv(C.T @ C) @ C.T @ y
##    E = y - C @ O_array #E@E
##    O_0 = (sum([c[1] for c in row]) / N(row)) - ((O_array[0] / N(row)) * sum([c[0] for c in row]))
##    O_1 = (sum([c[0] * c[1] for c in row]) - ((sum([c[0] for c in row]) * sum([c[1] for c in row])) / N(row))) / ( sum([c[0]**2.0 for c in row]) - ((sum([c[0] for c in row])**2.0) / N(row)) )
##    return [float(O_0), O_1]

##def mnktest(row, m = 1):
##    C = np.array([[1.0] + c[:m] for c in row])
##    #C = np.array([[0.0 for _ in range(m+1)] for __ in range(len(row))])
##    y = np.array([c[len(c)-1] for c in row])
##    for c in C:
##        c[0] = 1.0
##    O_array = np.linalg.inv(C.T @ C) @ C.T @ y
##    O_array = O_array.tolist()
##    
##    y_string = 'y = ' + str(round(O_array[0], 4))
##    if len(O_array) > 1:
##        for i in range(1, len(O_array)):
##            if O_array[i] > 0.0:
##                y_string += '+' + str(round(O_array[i], 4)) + '*x' + str(i)
##            else:
##                y_string += str(round(O_array[i], 4)) + '*x' + str(i)
##    print('Линейная модель:', y_string)
##
##    y_array = []
##    for i in range(len(row)):
##        temp = y_string[4:]
##        for j in range(m):
##            temp = temp.replace('x' + str(j+1), str(row[i][j]))
##        y_array.append(eval(temp))
##    print('Вектор прогноза y^:', y_array)
##
##    E_array = []
##    for i in range(len(row)):
##        E_array.append(row[i][len(row[0])-1] - y_array[i])
##    print('Вектор остатков E:', E_array)
##
##    delta_array = []
##    for i in range(len(row)):
##        delta_array.append(row[i][len(row[0])-1] - (sum(row[i][len(row[0])-1] for i in range(len(row)))) / len(row))
##    print('Вектор отклонений delta:', delta_array)
##    print('y среднее:', (sum(row[i][len(row[0])-1] for i in range(len(row)))) / len(row))#y среднее
##
##    E_array = np.array(E_array)
##    delta_array = np.array(delta_array)
##    r_squared = 1.0 - ((E_array.T @ E_array) / (delta_array.T @ delta_array))
##    print('Коэффициент детерминации R^2:', r_squared)
##    
##    return O_array

def mnktest(row, m=None):
    if m is None:
        m = list(range(len(row[0]) - 1))
    C = np.array([[1.0] + [row[i][j] for j in m] for i in range(len(row))])
    y = np.array([row[i][-1] for i in range(len(row))])

    for c in C:
        c[0] = 1.0

    O_array = np.linalg.inv(C.T @ C) @ C.T @ y
    O_array = O_array.tolist()

    y_string = 'y = ' + str(round(O_array[0], 4))
    if len(O_array) > 1:
        for i in range(1, len(O_array)):
            sign = '+' if O_array[i] > 0 else ''
            y_string += sign + str(round(O_array[i], 4)) + '*x' + str(i)
    print('Линейная модель:', y_string)

    y_pred_list = []
    for i in range(len(row)):
        temp_str = y_string[4:]
        for j_idx, j in enumerate(m):
            temp_str = temp_str.replace('x' + str(j_idx + 1), str(row[i][j]))
        y_pred_list.append(eval(temp_str))
    print('Вектор прогноза y^:', y_pred_list)

    E_array = [row[i][-1] - y_pred_list[i] for i in range(len(row))]
    print('Вектор остатков E:', E_array)

    mean_y = sum(row[i][-1] for i in range(len(row))) / len(row)
    delta_array = [row[i][-1] - mean_y for i in range(len(row))]
    print('Вектор отклонений delta:', delta_array)
    print('y среднее:', mean_y)

    E_array_np = np.array(E_array)
    delta_array_np = np.array(delta_array)
    r_squared = 1.0 - (E_array_np.T @ E_array_np) / (delta_array_np.T @ delta_array_np)
    print('Коэффициент детерминации R^2:', r_squared)

    return O_array

def pearson_correlation_matrix(data):
    """
    data: список списков, каждое внутреннее — набор переменных для одной точки, 
          например [[x1, x2, x3, y], ...].
    Возвращает матрицу корреляций в виде списка списков.
    """
    n_vars = len(data[0])
    n_samples = len(data)
    vars_list = [[] for _ in range(n_vars)]
    
    for row in data:
        for i in range(n_vars):
            vars_list[i].append(row[i])
    
    def mean(arr):
        return sum(arr) / len(arr)
    
    corr_matrix = [[0.0 for _ in range(n_vars)] for _ in range(n_vars)]
    
    for i in range(n_vars):
        for j in range(i, n_vars):
            X = vars_list[i]
            Y = vars_list[j]
            mean_X = mean(X)
            mean_Y = mean(Y)
            
            numerator = sum((X[k] - mean_X) * (Y[k] - mean_Y) for k in range(n_samples))
            denominator_x = sum((X[k] - mean_X)**2 for k in range(n_samples))
            denominator_y = sum((Y[k] - mean_Y)**2 for k in range(n_samples))
            
            denominator = (denominator_x * denominator_y)**0.5
            if denominator == 0:
                corr = 0
            else:
                corr = numerator / denominator
            
            corr_matrix[i][j] = corr
            corr_matrix[j][i] = corr
    
    for i in range(n_vars):
        corr_matrix[i][i] = 1.0
    
    return corr_matrix

a = pearson_correlation_matrix(t_row)
#print('Матрица парных корреляций:')
#for c in a:
#    print(c)
#print()

#print('Коэффициенты линейной модели:', mnktest(t_row))


#############################################################################

#Часть 6.2
#Входные данные
table_row = [
    [4922.4, 23369, 1865.9, 24951.2, 12.7, 669.4, 2404.8],
    [4130.7, 26629, 1807.9, 14648.1, 10.7, 644.1, 2302.2],
    [4137.4, 29792, 1749.5, 39558.7, 10.8, 668, 2206.2],
    [3889.4, 32495, 1690, 32365, 11.3, 693.7, 2190.6],
    [4263.9, 34030, 1577, 46568.8, 13.4, 611.6, 2388.5],
    [4243.5, 36709, 1444.5, 27929.6, 13.2, 608.3, 2160.1],
    [3966.5, 39167, 1304.6, 37218.5, 12.9, 611.4, 2058.5],
    [3657, 43724, 1208.6, 51418.1, 12.6, 583.9, 1991.5],
    [3461.2, 47867, 1126.7, 116166.5, 12.3, 620.7, 2024.3],
    [4316, 51344, 1102.8, 126304.8, 12.1, 564.7, 2044.2],
    [3624.6, 57244, 1077.7, 159875.4, 11, 644.2, 2004.4]]

#1.
#######################################
print(mnktest(table_row))

#2.
#######################################
#Анализ определителя матрицы корреляций
a = pearson_correlation_matrix(table_row)[:-1]
for sublist in a:
    sublist.pop()
print('Матрица парных корреляций:')
for c in a:
    print(c)
np_matrix = np.array(a)
det = np.linalg.det(np_matrix)
print('detRx =', "{:.10f}".format(det), 'Сильнаямультиколлинеарность факторов.')

#Анализ межкорреляций переменных
data = np.array(table_row)

X = data[:, :-1]
y = data[:, -1]

correlations_with_output = []
for i in range(X.shape[1]):
    corr, p_value = pearsonr(X[:, i], y)
    correlations_with_output.append(corr)

print("Коэффициенты корреляции с выходной переменной:")
for idx, corr in enumerate(correlations_with_output):
    print(f"Переменная {idx + 1}: {corr:.3f}")

selected_indices = [i for i, corr in enumerate(correlations_with_output) if abs(corr) >= 0.5]
print(f"\nВыбранные переменные (по индексу, корреляция >= 0.5): {selected_indices}")

X_selected = X[:, selected_indices]

corr_matrix = np.corrcoef(X_selected, rowvar=False)

print("\nМатрица парных корреляций выбранных переменных:")
print(corr_matrix)

max_corr = np.max(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])))
print(f"\nМаксимальное абсолютное значение корреляции между факторами (кроме диагонали): {max_corr:.3f}")

if max_corr > 0.8:
    print("Обнаружена сильная мультиколлинеарность между факторами.")
else:
    print("Мультиколлинеарность между факторами не выявлена или не сильная.")

#3.
#######################################

#print(mnktest(table_row, [0, 2]))
