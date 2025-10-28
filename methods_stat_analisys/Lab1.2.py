import matplotlib.pyplot as plt
from math import *
import numpy as np
d_row = [] #Дискретный ряд
N = 0 #Общая сумма частот ряда
with open(r'Дискретный ряд.txt', 'r') as f:
    r = f.readlines()
    for i in range(len(r)):
        d_row.append(r[i].split(" "))
        d_row[i][0] = int(d_row[i][0])
        d_row[i][1] = int(d_row[i][1])
        d_row[i][2] = float(d_row[i][2])
    f.close

for i in range(len(d_row)):
    N += d_row[i][1]

def average(): #Средняя
    average = 0
    for i in range(0, len(d_row)):
        average += float(d_row[i][0]) * float(d_row[i][1])
    return average / N

def variance():
    mx = 0
    mx2 = 0
    with open(r'Дискретный ряд.txt', 'r') as f:
        l = f.readlines()
        for i in range(0, len(l)):
            mx += float(l[i].split()[0]) * float(l[i].split()[1])
            mx2 += (float(l[i].split()[0])**2) * float(l[i].split()[1])
    mx = mx / N
    mx2 = mx2 / N
    return mx2 - (mx**2)

def average_square_deviation(): #СКО
    mx = 0
    mx2 = 0
    for i in range(0, len(d_row)):
        mx += float(d_row[i][0]) * float(d_row[i][1])
        mx2 += (float(d_row[i][0])**2) * float(d_row[i][1])
    mx = mx / N
    mx2 = mx2 / N
    return sqrt(mx2 - (mx**2))

def mu(power = 3): #Мю центральный момент
    a = 0
    av = average()
    for i in range(len(d_row)):
        a += d_row[i][1] * ((d_row[i][0] - av)**power)
    a /= N
    return a

def assymetry(): #Ассиметрия
    return mu(3.0) / (average_square_deviation()**3.0)

def excess(): #Эксцесс
    return (mu(4.0) / (average_square_deviation()**4.0)) - 3.0

def sr(sig):
    lower_bound = average() - sig * sqrt(variance())
    upper_bound = average() + sig * sqrt(variance())

    count_in_interval = 0
    for i in range(len(d_row)):
        if lower_bound <= d_row[i][0] and d_row[i][0] <= upper_bound:
            count_in_interval += d_row[i][1]

    prop_in_interval = count_in_interval / N
    print(f"Доля значений в интервале {sig} [{lower_bound:.4f}; {upper_bound:.4f}]: {prop_in_interval:.4%}")
    return prop_in_interval
    
def tsr(d): #Правило трёх сигм
    s1 = sr(1)
    s2 = sr(2)
    s3 = sr(3)

    return abs(s1 - 0.683) <= d and abs(s2 - 0.954) <= d and abs(s3 - 0.997) <= d

def stat_dist_func(data):  
    fig, ax = plt.subplots()
    
    x = [data[i][0] for i in range(len(data)-1)]
    y = [data[0][1]] + [data[i][1] + data[i-1][1] for i in range(1, len(data)-1)]
    
    ax.plot(x, y, color='lightgreen', linewidth=0.7)
    plt.show()

    return 0
    

print(round(assymetry(), 2))
print(round(excess(), 2)) 
print(tsr(0.005))

stat_dist_func(d_row)
    
