#!/usr/bin/env python
import pylab as pl
import numpy as np
import Queue as q

from math import sqrt

# Question 1.1
bfSel1 = np.loadtxt('data/bodyfat.txt', skiprows=117, usecols=(1,4,7,8,9), ndmin=2)
bfSel2 = np.loadtxt('data/bodyfat.txt', skiprows=117, usecols=(1,8), ndmin=2)

def partition(inset, p=0.8):
    data = np.copy(inset)
    np.random.shuffle(data)

    N = data.shape[0]
    return (data[:int(N * p)], data[int(N * p):])

(sel1Train, sel1Test) = partition(bfSel1)
(sel2Train, sel2Test) = partition(bfSel2)

def extractT(data):
    return (data[:,0].reshape(-1,1), data[:,1:])

(sel1TrainT, sel1Train) = extractT(sel1Train)
(sel1TestT, sel1Test) = extractT(sel1Test)
(sel2TrainT, sel2Train) = extractT(sel2Train)
(sel2TestT, sel2Test) = extractT(sel2Test)

# basis function
def phi(x):
    a = [e for e in x]
    a.insert(0, 1)
    return a

def phiC(x):
    return np.array(phi(x)).reshape(-1, 1)

def designMatrix(dataset):
    return np.array( [ phi(x) for x in dataset ] )

designSel1 = designMatrix(sel1Train)
designSel2 = designMatrix(sel2Train)

def w_ml(Phi, t):
    return np.dot(np.linalg.pinv(Phi), t)

wMLSel1 = w_ml(designSel1, sel1TrainT)
wMLSel2 = w_ml(designSel2, sel2TrainT)

def y(x, w):
    return np.dot(w.T, phiC(x))

def rms(x, t, w):
    print x.shape, t.shape, w.shape
    (N, _) = x.shape
    return sqrt(1.0 / N * sum((t[n] - y(x[n], w))**2 for n in range(N)))

rmsSel1 = rms(sel1Test, sel1TestT, wMLSel1)
rmsSel2 = rms(sel2Test, sel2TestT, wMLSel2)

# Question1.2
def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-mu)**2)/(2*sigma**2))

def mv_gauss(x, mu, sigma, sigma_inv):
    try:
        exponent = -(0.5)*(np.dot((x-mu).T,np.dot(sigma_inv, (x-mu))))
    except Exception as e:
        print x, mu, sigma, sigma_inv, e
    denominator = (2.0*np.pi)**(1.5) * np.linalg.norm(sigma)**(0.5)
    return 1/denominator * np.exp(exponent)

def MAP(Phi, t, alpha, beta=1):
    n = Phi.shape[1] # get size of matrix, i.e. number of columns
    S_N_inv = alpha*np.identity(n) + beta*np.dot(Phi.T, Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * np.dot(np.dot(S_N,Phi.T),t)
    return (m_N, S_N)

m_N1, S_N1 = MAP(designSel1, sel1TrainT, 0.1)
m_N2, S_N2 = MAP(designSel2, sel2TrainT, 0.1)
rmsSel1 = rms(sel1Test, sel1TestT, m_N1)
rmsSel2 = rms(sel2Test, sel2TestT, m_N2)
pl.figure()
pl.plot(bfSel2[:,1], bfSel2[:,0], 'ro')
xs = np.mgrid[80:150:100j]
y1 = np.array([y([x],wMLSel2) for x in xs]).reshape(-1)
y2 = np.array([y([x],m_N2) for x in xs]).reshape(-1)
pl.plot(xs,y1,'g-')
pl.plot(xs,y2,'b-')
pl.show()


# Question 2.1
irisTrain = np.loadtxt('data/irisTrain.dt', ndmin=2)
irisTest = np.loadtxt('data/irisTest.dt', ndmin=2)

pl.figure()
def plotGroup(data, n, color):
    points = [(r[0], r[1]) for r in data if r[2] == n]
    if len(points) == 0:
        # Prevent error if no points are misclassified in a group
        return
    ls, ws = zip(*points)
    pl.plot(ls, ws, color)

plotGroup(irisTrain, 0, 'yo')
plotGroup(irisTrain, 1, 'bo')
plotGroup(irisTrain, 2, 'ro')

pl.show()

# TODO: LDA

# Question 2.2

def norm(x):
    return np.sqrt(sum([xi**2 for xi in x]))

def distance(x,y,norm):
    return norm(x-y)

def mnorm(x):
    M = np.array([1, 0, 0, 10]).reshape(2,2)
    return norm(np.dot(M,x))

def kNN(x,k,norm): # assumes data in col 0,1, category in col 2
    nearest = q.PriorityQueue()
    for y in irisTrain:
        d = distance(x[0:2],y[0:2],norm)
        nearest.put((d, y[2]))

    closecats = [nearest.get()[1] for y in range(k)]
    return max(set(closecats), key=closecats.count)

def plot_knn(k,norm):
    group = []
    for x in irisTest:
        category = kNN(x,k,norm)
        group.append((x[0],x[1],(category, category == x[2])))

    print "kNN failures for k = ", k, ": ", len([1 for x in group if not x[2][1]])

    pl.figure()
    plotGroup(irisTrain, 0, 'y^')
    plotGroup(irisTrain, 1, 'b^')
    plotGroup(irisTrain, 2, 'r^')
    plotGroup(group, (0, True), 'yo')
    plotGroup(group, (1, True), 'bo')
    plotGroup(group, (2, True), 'ro')
    plotGroup(group, (0, False), 'yx')
    plotGroup(group, (1, False), 'bx')
    plotGroup(group, (2, False), 'rx')
    pl.show()

for k in [1,3,5,7]: plot_knn(k,norm)
for k in [1,3,5,7]: plot_knn(k,mnorm)
