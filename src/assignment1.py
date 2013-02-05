#!/usr/bin/env python
import pylab as pl
import numpy as np
from PIL import Image

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-mu)**2)/(2*sigma**2))

# Question 1.1
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
zlist = [pl.randn(2,1) for i in range(100)]
L = np.linalg.cholesky(sigma)
samples = [mu + L*z for z in zlist]
#print "samples:\n", samples


## Question 1.3
mu_estimate    = sum(samples) / len(samples)
sigma_estimate = sum([(x - mu_estimate)*(x - mu_estimate).T for x in samples]) / len(samples)

print "mu:", mu_estimate
print "sigma:", sigma_estimate.shape

xsamples, ysamples = [x[0][0,0] for x in samples], [x[1][0,0] for x in samples]
pl.plot(xsamples, ysamples,'ro', mu[0], mu[1], 'gv', mu_estimate[0], mu_estimate[1],'bo')
##TODO: Quantify in another way?
print "Sample mean deviation:", (mu - mu_estimate)[0,0]
pl.show()

## Question 1.4
## TODO

## Question 1.5
_, axes = pl.subplots(2, 4, sharex=True, sharey='row')
axes[0,0].hist(xsamples, normed=True, bins=5)
axes[0,1].hist(xsamples, normed=True, bins=10)
axes[0,2].hist(xsamples, normed=True, bins=20)
axes[0,3].hist(xsamples, normed=True, bins=30)
axes[1,0].hist(ysamples, normed=True, bins=5)
axes[1,1].hist(ysamples, normed=True, bins=10)
axes[1,2].hist(ysamples, normed=True, bins=20)
axes[1,3].hist(ysamples, normed=True, bins=30)

pl.title('Question 1.5')
pl.show()


# Question 1.9
im = Image.open('kande1.jpg').load()
training_set = [im[x,y] for x in range(150,330) for y in range(264,328)]
