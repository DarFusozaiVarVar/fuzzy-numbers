import matplotlib.pyplot as plt
from math import *
import numpy as np
#9 элементов в группе
#59 уникальных значений
#14-23, 23-32, 32-41, 41-50, 50-59, 59-68, 68-77
#n = 7
with open(r'Москва_2021.txt') as f:
    s = f.read().splitlines()
unique = []
unc = []
size = len(s)

with open(r'Дискретный ряд.txt', 'w+') as t:
    for i in s:
        if i not in unique:
            unique.append(i)
    unique.sort()
    for i in unique:
        unc.append(s.count(i))
    for i in range(0, len(unique)):
        #t.write(str(unique[i]) + ' ' + str(unc[i] / size) +'\n')
        t.write(str(unique[i]) + ' ' + str(unc[i]) + ' ' + str(unc[i] / size) + '\n')
    t.close()

n = 7 #Кол-во групп
min_number = int(unique[0])
max_number = int(unique[len(unique) - 1])
h = ceil((max_number - min_number) / n)
rows = [[min_number + i * h, 0] for i in range(n + 1)]

while len(unc) < n*h:
    unc.append(0)
for i in range(n):
    for j in range(h):
        rows[i][1] += unc[i * h + j]
with open(r'Интервальный ряд.txt', 'w+') as t:
    for i in range(0, len(rows) - 1):
        t.write(str(rows[i][0]) + '-' + str(rows[i+1][0]) + ' ' + str(rows[i][1]) + '\n')
    t.close()

def average():
    average = 0
    with open(r'Дискретный ряд.txt', 'r') as f:
        l = f.readlines()
    for i in range(0, len(l)):
        average += float(l[i].split()[0]) * float(l[i].split()[1])
    return average / size

def variance():
    mx = 0
    mx2 = 0
    with open(r'Дискретный ряд.txt', 'r') as f:
        l = f.readlines()
        for i in range(0, len(l)):
            mx += float(l[i].split()[0]) * float(l[i].split()[1])
            mx2 += (float(l[i].split()[0])**2) * float(l[i].split()[1])
    mx = mx / size
    mx2 = mx2 / size
    return mx2 - (mx**2)

def average_square_deviation():
    return sqrt(variance())

def coefficient_of_variation():
    mx = 0
    with open(r'Дискретный ряд.txt', 'r') as f:
        l = f.readlines()
    for i in range(0, len(l)):
        mx += float(l[i].split()[0]) * float(l[i].split()[1])
    mx = mx / size
    return average_square_deviation() / mx

def moda():
    m = 0
    max_number = 0
    with open(r'Дискретный ряд.txt', 'r') as f:
        l = f.readlines()
    for i in range(0, len(l)):
        if int(l[i].split()[1]) > m:
            m = int(l[i].split()[1])
            max_number = l[i].split()[0]
    return str(max_number) + " " + str(m)

def median():
    cumulative = 0
    for i in range(len(unique)):
        cumulative += unc[i]
        if cumulative == size / 2:
            next_index = unique.index(unique[i]) + 1
            if size % 2 == 0 and next_index < len(unique):
                return (float(unique[i]) + float(unique[next_index])) / 2
            else:
                return float(unique[i])
        elif cumulative > size / 2:
            return float(unique[i])

def i_average():
    average = 0
    for i in range(len(rows) - 1):
        average += ((rows[i][0] + rows[i+1][0]) / 2) * rows[i][1]
    return average / size

def i_variance():
    mx = i_average()
    variance = 0
    for i in range(len(rows) - 1):
        a = (rows[i][0] + rows[i+1][0]) / 2
        b = ((a - mx)**2) * rows[i][1]
        variance += b
        #variance += rows[i][1] * ((((rows[i][0] + rows[i+1][0]) / 2) - mx)**2)
    return variance / size

def i_average_square_deviation():
    return sqrt(i_variance())

def i_coefficient_of_variation():
    return (i_average_square_deviation() / i_average()) * 100

def i_moda():
    imoda = 0
    left_number = 0
    right_number = 0
    max_row = max(rows, key=lambda x: x[1])
    middle_number = max_row[1]
    left = max_row[0]
    if max_row[0] < rows[len(rows) - 1][0]:
        for i in range(len(rows) - 1):
            if max_row[0] == rows[i][0]:
                right = rows[i+1][0]
                right_number = rows[i+1][1]
                if max_row[0] != rows[0][0]:
                    left_number = rows[i-1][1]
    imoda = left + (right-left) * ((middle_number-left_number) / ((middle_number-left_number)+(middle_number-right_number)))
    return str(left) + "-" + str(right) + " " + str(imoda)

def i_median():
    imedian = 0
    summup = 0
    summup_before = 0
    for i in range(len(rows)):
        summup += rows[i][1]
        if summup > size/2:
            max_row = rows[i]
            left = rows[i][0]
            if rows[i] != rows[len(rows) - 1]:
                right = rows[i+1][0]
            else:
                right = 0
            break
    summup_before = summup - max_row[1]
    imedian = left + ((right-left) * (((size/2) - summup_before) / max_row[1]))
    return imedian
    
    

#Для Дискретного ряда
print("Для дискреционного ряда:")
print("Средняя", round(average(), 2)) #Средняя
print("Дисперсия", round(variance(), 2)) #Дисперсия
print("Среднее квадратическое отклонение", round(average_square_deviation(), 2)) #Среднее квадратическое отклонение
print("Коэффициент вариации", round(coefficient_of_variation(), 2)) #Коэффициент вариации
print("Мода", moda()) #Мода
print("Медиана", round(median(), 2)) #Медиана
print("Максимальное значение", max(unique)) #Максимальное значение
print("Минимальное значение", min(unique)) #Минимальное значение
print("Размах", round(float(max(unique)) - float(min(unique)), 2)) #Размах

print("")

#Для Интервального ряда
print("Для интервального ряда:")
print("Средняя", round(i_average(), 2)) #Средняя
print("Дисперсия", round(i_variance(), 2)) #Дисперсия
print("Среднее квадратическое отклонение", round(i_average_square_deviation(), 2)) #Среднее квадратическое отклонение
print("Коэффициент вариации", round(i_coefficient_of_variation(), 2)) #Коэффициент вариации
print("Мода", i_moda()) #Мода
print("Медиана", round(i_median(), 2)) #Медиана
print("Максимальное значение", rows[len(rows)-1][0]) #Максимальное значение
print("Минимальное значение", rows[0][0]) #Минимальное значение
print("Размах", rows[len(rows)-1][0] - rows[0][0]) #Размах

fig, (ax1, ax2) = plt.subplots(2)
ax1.grid(True, linestyle='-.')

while len(unc) != len(unique):
    unc.pop()
ax1.plot(unique, unc)
ax1.set_title('Дискретный ряд')

data = [rows[i][1] for i in range(len(rows)-1)]
bins = np.arange(rows[0][0], rows[len(rows)-1][0]+n, h)
interval_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]
ax2.bar(interval_labels, data, width=1, color='lightgreen', edgecolor="black", linewidth=0.7)
ax2.set_title('интервальный ряд')
plt.show()
