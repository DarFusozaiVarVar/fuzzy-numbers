from StatAnalysis import *
from math import *
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import chi2

with open(r"Интервальный ряд.txt", "r") as f:
    i_row = [
        [[int(p) for p in c.split(" ")[0].split("-")], int(c.split(" ")[1][2::])]
        for c in f.readlines()
    ]

with open(r"Москва_2021.txt", "r") as f:
    i_a_row = create_mean_interval_series([int(l.strip()) for l in f])

def fz(x, y=0.0, z=1.0):
    f = []
    for i in range(len(x)):
        f.append(1 / (z * sqrt(2*pi))) * exp(- ((x[i]-y)**2.0)/(2.0*(z**2.0)) )
    return f
    
##def i_chi_observable(row, n): #Хи наблюдаемое
##    n1 = []
##    n2 = []
##    
##    n1.append(row[0][1] + row[1][1])
##    for i in range(2, len(row)-2):
##        n1.append(row[i][1])
##    n1.append(row[len(row)-2][1] + row[len(row)-1][1])
##
##    n2.append(n[0]+n[1])
##    for i in range(2, len(n) - 2):
##        n2.append(n[i])
##    n2.append(n[len(n)-2] + n[len(n)-1])
##
##    return sum(((n1[i] - n2[i])**2) / n2[i] if n2[i] != 0.0 else float('inf') for i in range(len(n1)))
    

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
    if i == 0:
        zi.append((i_row[i][0][0] - i_sample_mean(i_row)) / i_standard_deviation(i_row))
    zi.append((i_row[i][0][1] - i_sample_mean(i_row)) / i_standard_deviation(i_row))
    #zi.append((xi[i]-i_sample_mean(i_row)) / i_standard_deviation(i_row)) #Это вроде не работает, сверху всё сверяется.
print("Стандартизированные значения xi", zi)

zi_intervals = [] #Интервалы zi, то есть [ [zi, zi+1], [zi, zi+1]... ]
for i in range(len(zi)-1):
    zi_intervals.append([zi[i], zi[i+1]])
#print(zi_intervals)

fzi = [] #Значения функции плотности нормального распределения
zi[0] = -float("inf")
zi[len(zi)-1] = float("inf")
cdf_values = norm.cdf(zi)
target_min, target_max = -0.5, 0.5 #Масштабирование в диапазон [-0.5, 0.5]
fzi = cdf_values * (target_max - target_min) + target_min
#for i in range(len(i_row)):
    #fzi.append(float(norm.pdf(zi[i], loc=0, scale=1)))
    #fzi.append( 1/sqrt(2*pi) * exp(-(zi[i]**2) / 2) ) #Это вроде не работает, сверху всё впринципе сверяется с погрешностями.
print("Значения функции плотности нормального распределения", fzi)

pi = [] #Вероятности pi попадания X в интервалы
for i in range(len(fzi)-1):
    pi.append(float(fzi[i+1] - fzi[i]))
print(pi)

ni = [] #Теоретические частоты
for i in range(len(pi)):
    ni.append(N(i_row)*pi[i])
##for i in range(len(i_row)):
##    ni.append((1*N(i_row) * fzi[i]) / i_standard_deviation(i_row))
print("Теоретические частоты ni", ni)

chi_obs = [] #Хи наблюдаемое
for i in range(len(ni)):
    chi_obs.append( ((i_row[i][1] - ni[i])**2.0) / ni[i] )
print("Значения хи наблюдаемого", chi_obs)

chi_crit = 9.487729
#k = len(i_row) - 2 - 2 - 1 #k-критерий k = m - r - 1
#chi_crit = chi2.ppf(0.05, k) #Хи критическое


#######################################################

print("Хи наблюдаемое:", sum(chi_obs), "Хи критическое:", chi_crit)
print("Генеральная совокупность распределена нормально" if chi_crit > sum(chi_obs) else "Генеральная совокупность не распределена нормально")
