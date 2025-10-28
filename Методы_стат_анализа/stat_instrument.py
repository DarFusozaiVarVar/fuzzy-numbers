from math import *
import random
import copy
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

class Row:

    row = []
    name = 'Untitled'

    def __init__(self, row, name = 'Untitled'):
        self.row = row
        self.name = name

class Discrete (Row):
    def __init__(self, row, name='Untitled'):
        super().__init__(row, name)
