from math import *
import random
import copy
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def average(row):
    average = 0
    l = copy.deepcopy(row)
    for i in range(0, len(l)):
        average += float(l[i][0]) * float(l[i][1])
    return average / N

def variance(row):
    mx = 0
    mx2 = 0
    l = copy.deepcopy(row)
    
    for i in range(0, len(l)):
        mx += float(l[i][0]) * float(l[i][1])
        mx2 += (float(l[i][0])**2) * float(l[i][1])
    mx = mx / N
    mx2 = mx2 / N
    return mx2 - (mx**2)

def i_average(row, w ,size):
    average = 0
    for i in range(len(row) - 1):
        average += ((row[i] + row[i+1]) / 2) * w[i]
    return average

def i_variance(row, w, size):
    mx = i_average(row, w, size)
    variance = 0
    for i in range(len(row) - 1):
        a = (row[i] + row[i+1]) / 2
        b = ((a - mx)**2) * w[i]
        variance += b
    return variance

def i_average_square_deviation(row, w, size):
    return sqrt(i_variance(row, w, size))

def sample_size(row, accuracy=3.0, reliability=0.95):
    z = stats.norm.ppf((1 + reliability) / 2)
    sigma = np.sqrt(variance(row))
    n = (z * sigma / accuracy)**2
    return ceil(n)

def generate_samples(row, num_samples=36):
    n = sample_size(row)
    data = []
    for age, freq, _ in row:
        data.extend([age] * freq)
    
    sample_means = []
    for _ in range(num_samples):
        sample = random.choices(data, k=n)
        sample_means.append(round(float(np.mean(sample)), 2))
    return sample_means

def gauss_curve(row, AVERAGE, ASD): #ASD stands for average square deviation
    out = []
    for i in range(0, len(row)):
        x = row[i]
        out.append(exp(-((x-AVERAGE)**2)/(2*ASD**2))/(ASD*sqrt(2*pi)))
    return out

d_row = [] #Дискретный ряд
N = 0 #Общая сумма частот ряда

with open(r'Москва_2021.txt') as f:
    s = f.read().splitlines()
unique = []
unc = []
N = len(s)

with open(r'Дискретный ряд.txt', 'w+') as t:
    for i in s:
        if i not in unique:
            unique.append(i)
    unique.sort()
    for i in unique:
        unc.append(s.count(i))
    for i in range(0, len(unique)):
        t.write(str(unique[i]) + ' ' + str(unc[i]) + ' ' + str(unc[i] / N) + '\n')
    t.close()

with open(r'Дискретный ряд.txt', 'r') as f:
    r = f.readlines()
    for i in range(len(r)):
        d_row.append(r[i].split(" "))
        d_row[i][0] = int(d_row[i][0])
        d_row[i][1] = int(d_row[i][1])
        d_row[i][2] = float(d_row[i][2])
    f.close


sample_means = generate_samples(d_row)
print(sample_means)

left_bound = floor(min(sample_means))
right_bound = ceil(max(sample_means))
intervals = list(range(left_bound, right_bound + 1))
freqs = [0] * (len(intervals) - 1)

for mean in sample_means:
    for i in range(len(intervals) - 1):
        if intervals[i] <= mean < intervals[i + 1]:
            freqs[i] += 1
            break
    else:
        if mean == intervals[-1]:
            freqs[-1] += 1

total = len(sample_means)
rel_freqs = [f / total for f in freqs]

with open(r'Интервальный ряд.txt', 'w', encoding='utf-8') as t:
    for i in range(len(freqs)):
        line = str(intervals[i]) + "-" + str(intervals[i + 1]) + " " + str(round(rel_freqs[i], 4)) + "\n"
        t.write(line)

print("Интервальные ряды и относительные частоты:")
for i in range(len(freqs)):
    print("[" + str(intervals[i]) + ", " + str(intervals[i+1]) + "): " + str(round(rel_freqs[i], 3)))

#График
fig, [ax1, ax2] = plt.subplots(2)
ax1.bar(
    [f"{intervals[i]}-{intervals[i+1]}" for i in range(len(freqs))],
    rel_freqs, width=0.8, align='center'
)
plt.xlabel("Интервалы")
plt.ylabel("Относительная частота")
plt.title("Интервальный ряд распределения выборочных средних")

#part 2
AVERAGE = i_average(intervals, rel_freqs, len(sample_means))
ASD = i_average_square_deviation(intervals, rel_freqs, len(sample_means))
print(AVERAGE)
print(ASD)
gauss_row = gauss_curve(intervals, AVERAGE, ASD)
print(gauss_row)
ax2.plot(intervals, gauss_row, color = 'red')
rf = copy.deepcopy(rel_freqs)
#rf.append(rf[-1])
#ax2.plot(intervals, rf, color = 'blue')
plt.show()

#gauss row test
#av = 0.168
#asd = 1.448
#fr = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
#print(gauss_curve(fr, av, asd))

#Выборочное среднее
mean_sample = np.mean(sample_means)

#Выборочная стандартной ошибки
s = np.std(sample_means, ddof = 1)

alpha = 1.0 - 0.95

#t критерий
t_crit = stats.t.ppf(1.0 - alpha / 2.0, len(sample_means) - 1.0)

#Доверительный интервал
lower_bound = mean_sample - t_crit * s / sqrt(len(sample_means))
upper_bound = mean_sample + t_crit * s / sqrt(len(sample_means))

print(f"Доверительный интервал с надежностью {int(0.95*100)}%: ({lower_bound:.2f}, {upper_bound:.2f})")
