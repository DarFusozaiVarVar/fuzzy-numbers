from StatAnalysis import *
from math import *
from scipy.stats import norm

with open(r"Москва_2021.txt", "r") as f:
    row = [int(c) for c in f.readlines()]
print(row)

# X - возраст Y - частота совершаемых преступлений

def linear_core_coff(X, Y): #функция для вычисления коэффицента линейной корреляции
    