#!/usr/bin/env python
import pylab as pl
import numpy as np
from PIL import Image
import random
from math import log, fabs

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-mu)**2)/(2*sigma**2))

## Question 1.1
pl.figure(1)
pl.title('Question 1.1')

pl.subplot(311)
x1 = np.mgrid[-5:5:100j]
value1 = gauss(x1,-1,1)
pl.xlabel('(mu,sigma) = (-1,1)')
pl.plot(x1, value1, 'r-')

pl.subplot(312)
x2 = np.mgrid[-5:5:100j]
value2 = gauss(x2,0,2)
pl.xlabel('(mu,sigma) = (0,2)')
pl.plot(x2, value2, 'b-')

pl.subplot(313)
x3 = np.mgrid[-10:10:100j]
value3 = gauss(x3,2,3)
pl.xlabel('(mu,sigma) = (2,3)')
pl.plot(x3, value3, 'g-')

pl.show()


## Question 1.2
sigma = np.mat("0.3 0.2; 0.2 0.2")
mu = np.mat("1;1")

def gen_samples(n):
    zlist = [pl.randn(2,1) for i in range(100)]
    L = np.linalg.cholesky(sigma)
    samples = [mu + L*z for z in zlist]
    xsamples, ysamples = [x[0][0,0] for x in samples], [x[1][0,0] for x in samples]
    return (samples, xsamples, ysamples)

samples, xsamples, ysamples = gen_samples(100)

## Question 1.3
mu_estimate    = sum(samples) / len(samples)
sigma_estimate = sum([(x - mu_estimate)*(x - mu_estimate).T for x in samples]) / len(samples)

print "mu:", mu_estimate
print "sigma:", sigma_estimate.shape

pl.plot(xsamples, ysamples,'ro', mu[0], mu[1], 'gv', mu_estimate[0], mu_estimate[1],'bo')
##TODO: Quantify in another way?
print "Sample mean deviation:", (mu - mu_estimate)[0,0]
pl.show()

## Question 1.4
## TODO

## Question 1.5
f, axes = pl.subplots(2, 4, sharex=True, sharey='row')
f.suptitle('Question 1.5')

axes[0,0].hist(xsamples, normed=True, bins=5)
axes[0,1].hist(xsamples, normed=True, bins=10)
axes[0,2].hist(xsamples, normed=True, bins=20)
axes[0,3].hist(xsamples, normed=True, bins=30)
axes[1,0].hist(ysamples, normed=True, bins=5)
axes[1,1].hist(ysamples, normed=True, bins=10)
axes[1,2].hist(ysamples, normed=True, bins=20)
axes[1,3].hist(ysamples, normed=True, bins=30)

pl.show()

## Question 1.6
pl.figure()
pl.title('Question 1.6')
xs = np.mgrid[-1:3:100j]
vs = gauss(xs, mu[0,0], sigma[0,0])

pl.hist(xsamples, normed=True, bins=30, color='g')
pl.plot(xs, vs, 'r-')

pl.show()

# \mu = 1, \sigma = 0.3
# p_1(x) = \frac{1}{0.3\sqrt{2\pi}} \exp \left\{ -\frac{1}{2 \cdot 0.3^2} (x - 1)^2 \right\}

## Question 1.7
ks, kxs, kys = gen_samples(1000)
tks, tkxs, tkys = gen_samples(10000)
f, axes = pl.subplots(2, 3)
f.suptitle('Question 1.7')

axes[0,0].hist2d(kxs, kys, bins=(10,10))
axes[0,1].hist2d(kxs, kys, bins=(15,15))
axes[0,2].hist2d(kxs, kys, bins=(20,20))
axes[1,0].hist2d(xsamples, ysamples, bins=(15,15))
axes[1,1].hist2d(kxs, kys, bins=(15,15))
axes[1,2].hist2d(tkxs, tkys, bins=(15,15))
pl.show()

## Question 1.8
def exp_transform(l, z):
    """ Transforms a uniformly distributed on the unit interval to an
        expontentially distributed value with parameter l.
        Inverse taken from p. 526, ex 11.2. """
    return -1/l * log(1 - z)

def exp_sample(l, n):
    """ n samples from the exponential distribution with parameter l. """
    return [exp_transform(l, random.random()) for i in range(n)]

print exp_sample(2, 10)
    
l = 2
true_mean = 1/l
cats = range(10, 1000, 100)
sample_means = [
    [ fabs(true_mean - sum(exp_sample(l, n))) for i in range(1000) ]
    for n in cats
]
pl.boxplot(sample_means)
pl.xticks(range(1, len(cats)+1, 1), cats)
pl.show()    
    
    
## Question 1.9
im = Image.open('kande1.jpg').load()
training_set = [im[x,y] for x in range(150,330) for y in range(264,328)]