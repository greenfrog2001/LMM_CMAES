import numpy as np

def Sphere(x):
    return sum(np.asarray(x)**2)


def SharpRidge(x):
    return x[0]**2 +100*np.sqrt(sum(np.asarray(x[1:])**2))


def Cigar(x):
    return x[0]**2 + 1e6*(sum(np.asarray(x[1:])**2))