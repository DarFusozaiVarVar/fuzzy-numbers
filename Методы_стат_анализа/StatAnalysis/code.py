from StatAnalysis import *
from math import *
from scipy.stats import norm
from scipy.stats import chi2

with open("Интервальный ряд.txt", "r") as f:
    i_row = [
        [[int(p) for p in c.split(" ")[0].split("-")], int(c.split(" ")[1])]
        for c in f.readlines()
    ]

def i_chi_observable(row, n): #Хи наблюдаемое
    n1 = []
    n2 = []
    
    n1.append(row[0][1] + row[1][1])
    for i in range(2, len(row)-2):
        n1.append(row[i][1])
    n1.append(row[len(row)-2][1] + row[len(row)-1][1])

    n2.append(n[0]+n[1])
    for i in range(2, len(n) - 2):
        n2.append(n[i])
    n2.append(n[len(n)-2] + n[len(n)-1])

    return sum(((n1[i] - n2[i])**2) / n2[i] if n2[i] != 0.0 else float('inf') for i in range(len(n1)))
    

xi = [(c[0][0] + c[0][1]) / 2 for c in i_row] #Средние xi

xini = [] #xi перемноженные на частоты
for i in range(len(i_row)):
    xini.append(xi[i] * i_row[i][1])

xini2 = [] #xi в квадрате перемноженные на частоты
for i in range(len(i_row)):
    xini2.append(xi[i]*xi[i]*i_row[i][1])

#i_sample_mean(i_row) Выборочная средняя

#i_standard_deviation(i_row) СКО

zi = [] #Стандартизированные значения xi
for i in range(len(i_row)):
    zi.append((xi[i]-i_sample_mean(i_row)) / i_standard_deviation(i_row))

fzi = [] #Значения функции плотности нормального распределения
for i in range(len(i_row)):
    fzi.append(float(norm.pdf(zi[i], loc=0, scale=1)))
    #fzi.append( 1/sqrt(2*pi) * exp(-(zi[i]**2) / 2) )

ni = [] #Теоретические частоты
for i in range(len(i_row)):
    ni.append((1*N(i_row) * fzi[i]) / i_standard_deviation(i_row))

k = len(i_row) - 2 - 2 - 1 #k-критерий k = m - r - 1

chi_crit = chi2.ppf(0.05, k) #Хи критическое

print(i_chi_observable(i_row, ni), chi_crit)
print("Генеральная совокупность распределена нормально" if chi_crit > i_chi_observable(i_row, ni) else "Генеральная совокупность не распределена нормально")
