from StatAnalysis import *
from math import *
import numpy as np
from scipy.stats import pearsonr
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy
with open("Москва_2021.txt", "r") as f:
    d_row = create_discrete_series([int(l.strip()) for l in f])

print('Часть 6.1\n')
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

def mnktest(row, m=None, print_results=False):
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
    if print_results:
        print('Линейная модель:', y_string)

    y_pred_list = []
    for i in range(len(row)):
        temp_str = y_string[4:]
        for j_idx, j in enumerate(m):
            temp_str = temp_str.replace('x' + str(j_idx + 1), str(row[i][j]))
        y_pred_list.append(eval(temp_str))
    if print_results:
        print('\nВектор прогноза y^:', y_pred_list)

    E_array = [row[i][-1] - y_pred_list[i] for i in range(len(row))]
    if print_results:
        print('\nВектор остатков E:', E_array)

    mean_y = sum(row[i][-1] for i in range(len(row))) / len(row)
    delta_array = [row[i][-1] - mean_y for i in range(len(row))]
    if print_results:
        print('\nВектор отклонений delta:', delta_array)
        print('\ny среднее:', mean_y)

    E_array_np = np.array(E_array)
    delta_array_np = np.array(delta_array)
    r_squared = 1.0 - (E_array_np.T @ E_array_np) / (delta_array_np.T @ delta_array_np)
    if print_results:
        print('\nКоэффициент детерминации R^2:', r_squared)

    return {'O_array': O_array,
            'y_string': y_string,
            'y_pred_list': y_pred_list,
            'E_array': E_array,
            'delta_array': delta_array,
            'mean_y': mean_y,
            'r_squared': r_squared}

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

def regression_analysis_scipy(row, degree):
    data = np.array(row)

    # Первый столбец - это x, последний - y
    X = data[:, 0]
    y = data[:, -1]

    # Подбираем коэффициенты многочлена степени degree
    coeffs = np.polyfit(X, y, degree)

    # Создаем функцию многочлена для оценки
    p = np.poly1d(coeffs)

    # Предсказания по модели
    y_pred = p(X)

    # Вычисляем R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - ss_res / ss_tot

    # Формируем уравнение в читаемом виде
    # coeffs — от старшей степени к младшей
    equation_terms = []
    degree_est = degree
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) > 1e-12:  # чтобы не выводить нулевые члены
            equation_terms.append(f"({c:.4f})*x^{power}")
    equation_str = "y = " + " + ".join(equation_terms)
    if not equation_terms:
        equation_str = "y = 0"

    print(f"Модель степени {degree}:")
    print("Уравнение:", equation_str)
    print("Коэффициент детерминации R^2:", R2)
    print()

    return {
        'coeffs': coeffs,
        'R2': R2,
        'equation': equation_str,
        'model': p
    }

a = pearson_correlation_matrix(d_row)
print('Матрица парных корреляций:')
for c in a:
    print(c)
print()

mnktest(d_row, print_results=True)

result_deg3 = regression_analysis_scipy(d_row, 3)
result_deg5 = regression_analysis_scipy(d_row, 5)

np_d_row = np.array(d_row)

X_data = np_d_row[:, 0]
y_data = np_d_row[:, -1]

#Диапазон X для гладких линий
X_plot = np.linspace(min(X_data), max(X_data), 500)
#Предсказания моделей
y_pred_deg3 = result_deg3['model'](X_plot)
y_pred_deg5 = result_deg5['model'](X_plot)
#Построение
plt.figure(figsize=(10, 6))
plt.scatter(X_data, y_data, color='blue', label='Исходные данные')
plt.plot(X_plot, y_pred_deg3, color='red', linewidth=2, label='Модель степени 3')
plt.plot(X_plot, y_pred_deg5, color='green', linewidth=2, label='Модель степени 5')
plt.xlabel('X (Возраст)')
plt.ylabel('Y (Число преступлений)')
plt.title('Зависимость с моделями полиномов')
plt.legend()
plt.grid(True)
plt.show()


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
print('\n\n\n\nЧасть 6.2\n')
#1.
#######################################
print(mnktest(table_row, print_results=True))

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
print('detRx =', "{:.10f}".format(det), 'Сильная мультиколлинеарность факторов.')

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

def func_1(row, m):
    Sfact = 0.0
    Se = 0.0
    y_ = mnktest(row)['y_pred_list']
    mean_y = mnktest(row)['mean_y']
    for i in range(len(row)):
        Sfact += (y_[i] - mean_y)**2.0
    for i in range(len(row)):
        Se += (row[i][len(row[0])-1] - y_[i])**2.0
    Sfact = Sfact / m
    Se = Se / (len(row) - m - 1)
    F = Sfact / Se
    p = stats.f.sf(Sfact / Se, m, len(row) - m - 1)
    return {'Sfact': Sfact,
            'Se': Se,
            'F': F,
            'p': p}

print(func_1(table_row, 6))

def func_2(row):
    n_samples = len(row)
    X = np.array([r[:-1] for r in row])
    y = np.array([r[-1] for r in row])
    p = X.shape[1] + 1
    
    X_with_const = np.column_stack((np.ones(n_samples), X))
    
    beta_hat = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
    
    y_pred = X_with_const @ beta_hat
    residuals = y - y_pred
    
    df_residual = n_samples - p
    sigma_squared = (residuals @ residuals) / df_residual
    
    cov_beta = sigma_squared * np.linalg.inv(X_with_const.T @ X_with_const)
    
    s_j = np.sqrt(np.diag(cov_beta))
    
    T_j = beta_hat / s_j
    
    p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=df_residual)) for t in T_j]
    
    return {
        'theta_j': beta_hat.tolist(),
        's_j': s_j.tolist(),
        'T_j': T_j.tolist(),
        'p_value': p_values
    }

print(func_2(table_row))

def func_3(row):
    n_samples = len(row)
    p = len(row[0])  # число признаков + 1 для y
    # Разделение на X и y
    X = np.array([r[:-1] for r in row])  # признаки
    y = np.array([r[-1] for r in row])   # зависимая переменная
    
    # Добавляем свободный член
    X_with_const = np.column_stack((np.ones(n_samples), X))
    
    # Оценка коэффициентов МНК
    beta_hat = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
    
    # Предсказания
    y_pred = X_with_const @ beta_hat
    
    # Полная сумма квадратов
    SST = np.sum((y - np.mean(y))**2)
    # Остаточная сумма квадратов
    SSR = np.sum((y - y_pred)**2)
    # Объясненная сумма квадратов
    SSE = np.sum((y_pred - np.mean(y))**2)
    
    # Коэффициент детерминации R^2
    R2 = 1 - SSR / SST
    
    # Скорректированный R^2
    R2_adj = 1 - (SSR / (n_samples - p)) / (SST / (n_samples - 1))
    
    # Множественный коэффициент корреляции R
    R = np.sqrt(R2)
    if np.corrcoef(y, y_pred)[0, 1] < 0:
        R = -R
    
    # Стандартная ошибка модели s
    residuals = y - y_pred
    s = np.sqrt(np.sum(residuals**2) / (n_samples - p))
    
    return {
        'R2': R2,
        'R2_adj': R2_adj,
        'R': R,
        's': s
    }

print(func_3(table_row))

#4.
#######################################

def stepwise_regression(table_row, alpha=0.1, initial_exclusions=None):
    data = table_row.copy()
    n_samples = len(data)
    n_features = len(data[0]) - 1
    current_features = list(range(n_features))
    y_values = np.array([row[-1] for row in data])
    
    # Начальная корреляция
    correlations = [np.abs(np.corrcoef([row[i] for row in data], y_values)[0,1]) for i in range(n_features)]
    exclude_indices = np.argsort(correlations)[:2]
    if initial_exclusions:
        for ix in initial_exclusions:
            if ix in current_features:
                current_features.remove(ix)
    for ix in sorted(exclude_indices, reverse=True):
        if ix in current_features:
            current_features.remove(ix)
    
    iteration = 1

    while True:
        X = np.array([[row[i] for i in current_features] for row in data])
        y = y_values
        X_with_const = np.column_stack((np.ones(n_samples), X))
        beta_hat = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
        y_pred = X_with_const @ beta_hat
        residuals = y - y_pred
        df_residual = n_samples - len(beta_hat)
        sigma_squared = (residuals @ residuals) / df_residual
        cov_beta = sigma_squared * np.linalg.inv(X_with_const.T @ X_with_const)
        s_j = np.sqrt(np.diag(cov_beta))
        T_j = beta_hat / s_j

        # p-значения для коэффициентов (только для признаков, без константы)
        p_values_full = [2 * (1 - stats.t.cdf(np.abs(t), df=df_residual)) for t in T_j]
        p_vals = p_values_full[1:]  # исключая константу
        
        # Округляем p-значения до десятых
        p_vals_rounded = [round(p, 1) for p in p_vals]

        # Вывод
        print(f"\nШаг {iteration}")
        print(f"Текующие переменные: {current_features}")
        print(f"Коэффициенты: {beta_hat.tolist()}")
        print(f"p-значения: {p_vals_rounded}")

        # Проверка условия завершения
        if all(p <= alpha for p in p_vals_rounded):
            print("Условие завершения достигнуто.")
            break
        else:
            # Удаляем переменную с наибольшим p (после округления)
            max_p = max(p_vals_rounded)
            if max_p <= alpha:
                print("Условие завершения достигнуто.")
                break
            idx_max = p_vals_rounded.index(max_p)
            # Удаляем соответствующую переменную (учитывая, что p_vals_rounded без константы)
            del current_features[idx_max]
            if len(current_features) == 0:
                print("Все признаки исключены.")
                break
            iteration += 1

stepwise_regression(table_row, alpha=0.1)
