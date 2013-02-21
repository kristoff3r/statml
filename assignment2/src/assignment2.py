#!/usr/bin/env python
import numpy as np
import csv

# Question 1.1
bfT = np.loadtxt('../Data/bodyfat.txt', skiprows=117, usecols=(1,), ndmin=2)
bfSel1 = np.loadtxt('../Data/bodyfat.txt', skiprows=117, usecols=(4,7,8,9), ndmin=2)
bfSel2 = np.loadtxt('../Data/bodyfat.txt', skiprows=117, usecols=(8,), ndmin=2)

# basis function
def phi(x):
    a = [e for e in x]
    a.insert(0, 1)
    return a

def phiC(x):
    return np.array(phi(x)).reshape(-1, 1)

def designMatrix(dataset):
    return np.array( [ phi(x) for x in dataset ] )

designSel1 = designMatrix(bfSel1)
designSel2 = designMatrix(bfSel2)

def w_ml(Phi, t):
    return np.dot(np.linalg.pinv(Phi), t)

wMLSel1 = w_ml(designSel1, bfT)
wMLSel2 = w_ml(designSel2, bfT)

def y(x, w): return np.dot(w.T, phiC(x))

def rms(x, t, w):
    (N, _) = x.shape
    return 1.0 / N * sum((t[n] - y(x[n], w))**2 for n in range(N))

rmsSel1 = rms(bfSel1, bfT, wMLSel1)
rmsSel2 = rms(bfSel2, bfT, wMLSel2)

print rmsSel1
print rmsSel2
