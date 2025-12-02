from StatAnalysis import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
import copy
with open("Москва_2021.txt", "r") as f:
    d_row = create_discrete_series([int(l.strip()) for l in f])

test_row = [[13,19], [15,14], [18,15], [20, 10], [23,12], [26,8], [31,5]]

def structural_identification(row, print_table=False):
    x1 = row[0][0]
    y1 = row[0][1]
    xn = row[len(row)-1][0]
    yn = row[len(row)-1][1]
    x_mean_1 = round((x1 + xn) / 2.0, 1)
    y_mean_1 = (y1 + yn) / 2.0
    x_mean_2 = round(sqrt(x1 * xn), 1)
    y_mean_2 = sqrt(y1 * yn)
    x_mean_3 = round((2.0 * x1 * xn) / (x1 + xn), 1)
    y_mean_3 = (2.0 * y1 * yn) / (y1 + yn)
    xy_means_for_analysis = [[x_mean_1, y_mean_1], [x_mean_2, y_mean_2], [x_mean_1, y_mean_2], [x_mean_3, y_mean_1], [x_mean_1, y_mean_3], [x_mean_3, y_mean_3], [x_mean_2, y_mean_1]]
    xy_s = []
    sum_y = sum(item[1] for item in row)
    for x_mean, y_mean in xy_means_for_analysis:
        if [x_mean, y_mean] in row:
            ys = y_mean
            xy_s.append([x_mean, ys])
        else:
            for i in range(len(row)):
                if row[i][0] >= x_mean:
                    xi = row[i-1][0]
                    yi = row[i-1][1]
                    xi1 = row[i][0]
                    yi1 = row[i][1]
                    break
            ys = yi + ((yi1-yi)/(xi1-xi)) * (x_mean - xi)
            print(x_mean, ys)
            xy_s.append([x_mean, ys])
        
    result_deltas = []
    result_dependencies = ['y = ax + b', 'y = a * x^b', 'y = a * b^x', 'y = a + b/x', 'y = 1 / (ax + b)', 'y = x / (ax + b)', 'y = a * ln(x) + b']
    for i in range(len(xy_s)):
        result_deltas.append([abs(xy_means_for_analysis[i][1] - xy_s[i][1]), 100.0 * (abs(xy_means_for_analysis[i][1] - xy_s[i][1]) / sum_y)])

    if print_table:
        print('№ структура x_mean y_mean y_s delta_s delta_s%')
        for i in range(len(xy_s)):
            print(f'{i+1} {result_dependencies[i]} {round(xy_s[i][0], 2)} {round(xy_means_for_analysis[i][1], 2)} {round(xy_s[i][1], 2)} {round(result_deltas[i][0], 2)} {round(result_deltas[i][1], 2)}%')
        min_delta = float('inf')
        min_index = 0
        for i in range(len(result_deltas)):
            if result_deltas[i][0] < min_delta:
                min_delta = result_deltas[i][0]
                min_index = i
        print(f'min_delta = {round(min_delta, 2)} соответствует зависимости {result_dependencies[min_index]}')
    return "Не назначил, что конкретно выводить!"

def exponent_of_the_approximating_polynomial(row, percent=0.02, print_row=False):
    sum_y = sum(item[1] for item in row)
    temp_row = [item[1] for item in row]
    result_row = []
    delta_number = 1
    maxim = float('inf')
    if print_row:
        print(f'2% от суммы частот = {sum_y} * {percent} = {sum_y * percent}')
        print('Найденные последовательности разности:')
    while maxim > sum_y * percent:
        maxim = -float('inf')
        for i in range(len(temp_row) - 1):
            result_row.append(abs(temp_row[i] - temp_row[i+1]))

        for i in range(len(result_row)):
            if result_row[i] > maxim:
                maxim = result_row[i]
        temp_row = result_row
        result_row = []
        if print_row:
            print(f'delta = {delta_number}, макс = {maxim}')
        delta_number += 1
    return delta_number-1

structural_identification(d_row, True)
print('Показатель степени аппроксимирующего многочлена =', exponent_of_the_approximating_polynomial(d_row, 0.02, True))


####################################################################
#Часть 5.2
print('\tЧасть 5.2')
def covariation(row): #Ковариация
    covar = 0.0
    x_a = 0.0
    y_a = 0.0
    for i in range(len(row)):
        x_a += row[i][0] / len(row)
        y_a += row[i][1] / len(row)
    for i in range(len(row)):
        covar += ((row[i][0] - x_a) * (row[i][1] - y_a)) / len(row)
    return covar

def standard_deviations_for_x_y(row): #Стандартные отклонения (и средние) для x и y, оформленные в виде словаря.
    x_a = 0.0
    x_a2 = 0.0
    y_a = 0.0
    y_a2 = 0.0
    x_sd = 0.0
    y_sd = 0.0
    for i in range(len(row)):
        x_a += row[i][0] / len(row)
        x_a2 += (row[i][0] * row[i][0]) / len(row)
        y_a += row[i][1] / len(row)
        y_a2 += (row[i][1] * row[i][1]) / len(row)
    x_sd = sqrt(x_a2 - (x_a ** 2))
    y_sd = sqrt(y_a2 - (y_a ** 2))
    return {"среднняя X": x_a, "среднняя Y": y_a, "СКО X": x_sd, "СКО Y": y_sd}

def correlation_coefficient(row, print_result = False): #Коэффициент корреляции
    r_xy = covariation(row) / (standard_deviations_for_x_y(row)["СКО X"] * standard_deviations_for_x_y(row)["СКО Y"])
    if print_result:
        print_res = ""
        if 0 < abs(r_xy) <= 0.3:
            print_res = "Слабая "
            if r_xy < 0:
                print_res += "обратная "
            else:
                print_res += "прямая "
            print_res += "зависимость"
        elif 0.3 < abs(r_xy) <= 0.5:
            print_res = "Умеренная "
            if r_xy < 0:
                print_res += "обратная "
            else:
                print_res += "прямая "
            print_res += "зависимость"
        elif 0.5 < abs(r_xy) <= 0.7:
            print_res = "Заметная "
            if r_xy < 0:
                print_res += "обратная "
            else:
                print_res += "прямая "
            print_res += "зависимость"
        elif 0.7 < abs(r_xy) < 1:
            print_res = "Сильная "
            if r_xy < 0:
                print_res += "обратная "
            else:
                print_res += "прямая "
            print_res += "зависимость"
        elif abs(r_xy) == 1:
            print_res = "Функциональная"
            if r_xy < 0:
                print_res += "обратная "
            else:
                print_res += "прямая "
            print_res += "зависимость"
        print(print_res)
    return r_xy

def parametric_identification(row, print_results=False):
    row_squared = []
    for c in row:
        row_squared.append([c[0]**2, c[1]**2])
    
    row_xy = []
    for c in row:
        row_xy.append(c[0] * c[1])

    x_a = 0.0
    y_a = 0.0
    x2_a = 0.0
    y2_a = 0.0
    xy_a = 0.0
    for i in range(len(row)):
        x_a += row[i][0] / len(row)
        x2_a += (row[i][0] * row[i][0]) / len(row)
        y_a += row[i][1] / len(row)
        y2_a += (row[i][1] * row[i][1]) / len(row)
        xy_a += (row[i][0] * row[i][1]) / len(row)

    dx = x2_a - (x_a ** 2.0)
    b = (xy_a - (x_a * y_a)) / dx
    a = y_a - (b * x_a)

    if print_results:
        print(f'x средняя = {x_a}')
        print(f'y средняя = {y_a}')
        print(f'x^2 средняя = {x2_a}')
        print(f'y^2 средняя = {y2_a}')
        print(f'xy средняя = {xy_a}')
        print(f'Дисперсия D(X) = {dx}')
        print(f'Коэффициент b = {b}')
        print(f'Коэффициент a = {a}')
        
    return f"y = {round(a, 2)} + ({round(b, 2)}*x)"

def determination_coefficient(row):
    r_squared = 0.0
    numerator = 0.0
    denumerator = 0.0
    equation = parametric_identification(row)[4:]
    y_row = []
    for i in range(len(row)):
        y_row.append(eval(equation.replace('x', str(row[i][0]))))

    y_a = 0.0
    for i in range(len(row)):
        y_a += row[i][1] / len(row)

    for i in range(len(row)):
        numerator += (row[i][1] - y_row[i])**2
        denumerator += (row[i][1] - y_a)**2
    r_squared = 1.0 - (numerator/denumerator)
    return r_squared

print(parametric_identification(d_row, True))
print('R^2 =', determination_coefficient(d_row))
print(f'r_xy = {correlation_coefficient(d_row)}; sqrt(R^2) = {sqrt(determination_coefficient(d_row))}')


parametric_str = parametric_identification(d_row)[4:]

#Преобразование списка списков в numpy массив
test_row = np.array(d_row)

x_data = test_row[:, 0]
y_data = test_row[:, 1]

#Создание сетки
x_min, x_max = x_data.min(), x_data.max()
y_min, y_max = y_data.min(), y_data.max()
grid_size = 50
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x_grid, y_grid)

#Вычисление Z на сетке
Z = np.empty_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_val = X[i, j]
        try:
            # Передаем переменную x в eval
            Z[i, j] = eval(parametric_str, {"x": x_val})
        except Exception as e:
            print(f"Ошибка при eval в точке ({x_val}): {e}")
            Z[i, j] = np.nan

#Построение корреляционного поля
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=100, cmap='viridis')
plt.colorbar(label='Значение функции')

#Построение исходных точек (заполненными)
plt.scatter(x_data, y_data, facecolors='black', edgecolors='black', marker='o', label='Точка дискретного ряда')

#Построение аппроксимирующей прямой
x_fit = np.linspace(x_min, x_max, 200)
y_fit = [eval(parametric_str, {"x": x}) for x in x_fit]
plt.plot(x_fit, y_fit, color='black', linewidth=2, linestyle='--', label='Аппроксимирующая прямая')

#Обеспечение масштабируемости графика
plt.xlim(x_min - (x_max - x_min)*0.05, x_max + (x_max - x_min)*0.05)
plt.ylim(y_min - (y_max - y_min)*0.05, y_max + (y_max - y_min)*0.05)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Корреляционное поле и аппроксимирующая прямая')
plt.grid(True)
plt.show()

