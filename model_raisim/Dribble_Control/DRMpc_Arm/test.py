import numpy as np

def insert(x, n):
    x[0] = 3
    x = np.concatenate([x, [1]], axis = 0)
    n = 3
    print(x)
    return
def a():
    n = 1
    x = np.array([0, 1, 2])
    insert(x,n)
    print(x, n)

if __name__ == "__main__":
    a()