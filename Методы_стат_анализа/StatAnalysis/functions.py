from math import *
from scipy.stats import norm
import numpy as np
import random


i_test = [([14,23], 4811),
([23,32], 9476),
([32,41], 7243),
([41,50], 7951),
([50,59], 1140),
([59,68], 1472),
([68,77], 330)] #Интервальный ряд распределения для самопроверки


##### ОБЩИЕ ФУНКЦИИ РЯДОВ #####

def N(row): #Сумма частот N
    return sum(c[1] for c in row)

##### ФУНКЦИИ ДЛЯ ТЕОРЕТИЧЕСКИХ ДАННЫХ #####

##### ФУНКЦИИ ДИСКРЕТНЫХ РЯДОВ #####
##### row = [ [xi, ni], [xi, ni]... ] #####

def math_expect(row): #Математическое ожидание M(X)
    me = 0
    for i in range(0, len(row)):
        me += row[i][0] * (row[i][1] / N(row))
    return me

def dispersion(row): #Дисперсия D(X) (Не делённая на N)
    mx = 0
    mx2 = 0
    for i in range(len(row)):
        mx += row[i][0] * (row[i][1] / N(row))
        mx2 += row[i][0] * row[i][0] * (row[i][1] / N(row))
    return mx2 - (mx**2)

def standard_deviation(row): #Среднее квадратическое отклонение sigma(X) (От дисперсии, НЕ ДЕЛЁННОЙ НА N)
    return sqrt(dispersion(row))

def t_coeff_of_variation(row): #Теоретический коэффициент вариации V(X) равный (СКО/выборочная_средняя) * 100%
    return round((standard_deviation(row) / math_expect(row)) * 100, 2)

##### ФУНКЦИИ ИНТЕРВАЛЬНЫХ РЯДОВ #####
##### row = [ [[xi, xi+1], ni], [[xi, xi+1], ni]... ] #####

def i_math_expect(row): #Математическое ожидание M(X) (Не делённое на N)
    me = 0
    for i in range(0, len(row)):
        me += ((row[i][0][0] + row[i][0][1]) / 2) * (row[i][1] / N(row))
    return me

def i_dispersion(row): #Дисперсия D(X) (Не делённая на  N)
    mx = 0
    mx2 = 0
    for i in range(len(row)):
        mx += ((row[i][0][0] + row[i][0][1]) / 2) * (row[i][1] / N(row))
        mx2 += ((row[i][0][0] + row[i][0][1]) / 2) * ((row[i][0][0] + row[i][0][1]) / 2) * (row[i][1] / N(row))
    return mx2 - (mx**2)

def i_standard_deviation(row): #Среднее квадратическое отклонение sigma(X) (От дисперсии, НЕ ДЕЛЁННОЙ НА N)
    return sqrt(i_dispersion(row))

def t_i_coeff_of_variation(row): #Теоретический коэффициент вариации V(X) равный (СКО/выборочная_средняя) * 100%
    return round((i_standard_deviation(row) / i_math_expect(row)) * 100, 2)

##### ФУНКЦИИ ДЛЯ СТАТИСТИЧЕСКИХ ДАННЫХ #####

##### ФУНКЦИИ ДИСКРЕТНЫХ РЯДОВ #####
##### row = [ [xi, ni], [xi, ni]... ] #####

def sample_mean(row): #Выборочная средняя (x с чертой)
    mx = 0
    for i in range(len(row)):
        mx += row[i][0] * row[i][1]
    return mx / N(row)

def mode(row): #Мода M0
    max_freq = max(c[1] for c in row)
    for x, ni in row:
        if ni == max_freq:
            return x

def median(row): #Медиана Me
    cumulative = 0
    median_pos = N(row) / 2
    for value, frequency in row:
        cumulative += frequency
        if cumulative >= median_pos:
            return value
    return None

def scope(row): #Размах R
    return row[len(row)-1][0] - row[0][0]

def deviations(row): #Отклонения xi - x с чертой
    return [c[0] - sample_mean(row) for c in row]

def coeff_of_variation(row): #Коэффициент вариации V(X) равный (СКО/выборочная_средняя) * 100%
    return round((standard_deviation(row) / sample_mean(row)) * 100, 2)

def integral_distribution_function(row): #Интегральная функция распределения F(x) = P(X<x) Вторая лекция
    summa = 0
    t = [0.0]
    for i in range(len(row)):
        summa += row[i][1] / N(row)
        t.append(summa)
    return t

#scipy.stats.norm.pdf(x, mu, sigma) - Плотность распределения вероятностей, mu и sigma по умолчанию равны 0 и 1

def starting_moment(row, k = 0): #Начальный момент порядка k μk
    me = 0
    for i in range(len(row)):
        me += (row[i][0]**k) * row[i][1]
    return me / N(row)

def central_moment(row, k = 0): #Центральный момент порядка k μk
    cm = 0
    for i in range(len(row)):
        cm += (row[i][0] - math_expect(row))**k * row[i][1]
    return cm / N(row)

def asymmetry(row): #Асимметрия As
    return central_moment(row, 3) / (standard_deviation(row)**3)

def excess(row): #Эксцесс Ek
    return (central_moment(row, 4) / (standard_deviation(row)**4)) - 3

def sigma_rule(row, a=0, sigma=1, tolerance=0.05): #Правило трёх сигм
    total_N = sum(n_i for _, n_i in row)
    theory_probs = [0.683, 0.954, 0.997]
    intervals = [
        (a - sigma, a + sigma),
        (a - 2 * sigma, a + 2 * sigma),
        (a - 3 * sigma, a + 3 * sigma)]
    
    results = []
    for (left, right), theory_prob in zip(intervals, theory_probs):
        observed_count = 0
        for x_i, n_i in row:
            if left <= x_i <= right:
                observed_count += n_i
        observed_prob = observed_count / total_N
        results.append(abs(observed_prob - theory_prob) <= tolerance)
    return results

def get_sample_size(row, gamma=0.95, delta=3.0): #Определение объёма выборки
    t = norm.ppf(1 - (1 - gamma) / 2)
    return (t * standard_deviation(row) / delta)**2

def generate_samples(row, amount=36, gamma=0.95, delta=3.0):
    s = []
    n = int(np.ceil(get_sample_size(row, gamma, delta)))
    data = []
    for i in range(len(row)):
        data.extend([row[i][0]] * row[i][1])
    for _ in range(amount):
        generated = random.choices(data, k=n)
        generated = [int(round(val)) for val in generated]
        s.append(generated)
    return s

##### ФУНКЦИИ ИНТЕРВАЛЬНЫХ РЯДОВ #####
##### row = [ [[xi, xi+1], ni], [[xi, xi+1], ni]... ] #####

def i_sample_mean(row): #Выборочная средняя  (x с чертой)
    mx = 0
    for i in range(len(row)):
        mx += ((row[i][0][0] + row[i][0][1]) / 2) * row[i][1]
    return mx / N(row)

def i_mode(row): #Мода M0
    max_n = max(row, key=lambda x: x[1])[1]
    m = next(i for i, item in enumerate(row) if item[1] == max_n)
    
    l, u = row[m][0]
    h = u - l
    
    n_m = row[m][1]
    n_prev = row[m-1][1] if m > 0 else 0
    n_next = row[m+1][1] if m < len(row)-1 else 0
    
    if (n_m - n_prev) + (n_m - n_next) == 0:
        return l
    
    mode = l + h * (n_m - n_prev) / ((n_m - n_prev) + (n_m - n_next))
    return mode

def i_median(row): #Медиана Me
    total_freq = sum(interval[1] for interval in row)
    half_total = total_freq / 2

    cumulative = 0
    median_interval_index = None

    for i, (interval, freq) in enumerate(row):
        cumulative += freq
        if cumulative >= half_total:
            median_interval_index = i
            break

    if median_interval_index is None:
        return None

    l, u = row[median_interval_index][0]
    h = u - l
    f_m = row[median_interval_index][1]

    F_prev = cumulative - f_m
    N_over_2 = half_total

    median = l + h * (N_over_2 - F_prev) / f_m
    return median

def i_scope(row): #Размах R
    return row[len(row)-1][0][1] - row[0][0][0]

def i_coeff_of_variation(row): #Коэффициент вариации V(X) равный (СКО/выборочная_средняя) * 100%
    return int(i_standard_deviation(row) / i_sample_mean(row)) * 100

def i_integral_distribution_function(row): #Интегральная функция распределения F(x) = P(X<x) Вторая лекция
    summa = 0
    t = [0.0]
    s = []
    for i in range(len(row)):
        if row[i][0][0] not in s:
            s.append(row[i][0][0])
        if row[i][0][1] not in s:
            s.append(row[i][0][1])
    
    for i in range(len(s)-1):
        summa += row[i][1] / N(row)
        t.append(summa)
    print(s) #Все уникальные xi и xi+1 в ряду в порядке возрастания (без повторов). Может понадобятся
    return t

#scipy.stats.norm.pdf(x, mu, sigma) - Плотность распределения вероятностей, mu и sigma по умолчанию равны 0 и 1

def i_starting_moment(row, k = 0): #Начальный момент порядка k μk
    sm = 0
    for i in range(len(row)):
        sm += (((row[i][0][0] + row[i][0][1]) / 2)**k) * row[i][1]
    return sm / N(row)

def i_central_moment(row, k = 0): #Центральный момент порядка k μk
    cm = 0
    for i in range(len(row)):
        cm += (((row[i][0][0] + row[i][0][1]) / 2) - i_sample_mean(row))**k * row[i][1]
    return cm / N(row)

def i_asymmetry(row): #Асимметрия As
    return i_central_moment(row, 3) / (i_standard_deviation(row)**3)

def i_excess(row): #Эксцесс Ek
    return (i_central_moment(row, 4) / (i_standard_deviation(row)**4)) - 3

def i_sigma_rule(row, a=0, sigma=1, tolerance=0.05): #Правило трёх сигм
    total_N = N(row)
    intervals = [
        (a - sigma, a + sigma),
        (a - 2 * sigma, a + 2 * sigma),
        (a - 3 * sigma, a + 3 * sigma)]
    theory_probs = [0.683, 0.954, 0.997]
    results = []
    for (left, right), theory_prob in zip(intervals, theory_probs):
        observed_count = 0     
        for (x_i, x_next), n_i in row:
            if x_next > left and x_i < right:
                observed_count += n_i       
        observed_prob = observed_count / total_N
        results.append(abs(observed_prob - theory_prob) <= tolerance)
    return results

def i_get_sample_size(row, gamma=0.95, delta=3.0): #Определение объёма выборки
    t = norm.ppf(1 - (1 - gamma) / 2)
    return (t * i_standard_deviation(row) / delta)**2

def i_generate_samples(row, amount = 36, gamma = 0.95, delta = 3.0): #Генерирование выборок с заданным объёмом
    s = []
    data = []
    for i in range(len(row)):
        data.extend([((row[i][0][0] + row[i][0][0]) / 2)] * row[i][1])
    for i in range(amount):
        samples = []
        std_dev = i_standard_deviation(row)
        for j in range(len(row)):
            xi = row[j][0][0]
            ni = row[j][1]
            n_i = int(np.ceil(i_get_sample_size(row, gamma, delta)))
            generated = np.random.uniform(min([min(c[0][0], c[0][1]) for c in row]), max([max(c[0][0], c[0][1]) for c in row]), size=n_i)
            #generated = np.random.normal(loc=xi, scale=std_dev, size=n_j)
            samples.append(generated)
        s.append(np.array(samples))
    return s

#print(i_sigma_rule(i_test))
#print(i_generate_samples(i_test))
