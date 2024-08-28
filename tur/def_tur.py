import numpy as np


def plus_opertor(x, y):
    res = x + y
    return res

def minus_opertor(x, y):
    res = x - y
    return res

x = np.array([1, 2])
y = np.array([1, 2])

res = minus_opertor(x, y)
print(res)