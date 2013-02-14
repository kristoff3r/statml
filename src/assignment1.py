#!/usr/bin/env python
import pylab as pl
import numpy as np
from PIL import Image
import random
from math import log, fabs, sqrt

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-((x-mu)**2)/(2*sigma**2))

## Question 1.1
pl.figure()
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
print "sigma:", sigma_estimate

pl.plot(xsamples, ysamples,'ro', mu[0], mu[1], 'gv', mu_estimate[0], mu_estimate[1],'bo')
print "Sample mean deviation:", (mu - mu_estimate)[0,0]
pl.show()

## Question 1.5
f, axes = pl.subplots(2, 4, sharey='row')
f.suptitle('Question 1.5')

bins = [5,10,20,30]
varians = [0.3, 0.2]
samples = [xsamples, ysamples]

for i in range(2):
    for j in range(4):
        axes[i,j].hist(samples[i], normed=True, bins=bins[j])
        axes[i,j].set_title('bins = %d' % bins[j])
        x = np.mgrid[-1:3:100j]
        value = gauss(x,1,sqrt(varians[i]))
        axes[i,j].plot(x,value,'r-')

for foo in axes:
    for axis in foo:
        for label in axis.get_xticklabels():
            label.set_rotation(random.randrange(-90,90))

pl.show()

## Question 1.6
pl.figure()
pl.title('Question 1.6')
xs = np.mgrid[-1:3:100j]
vs = gauss(xs, mu[0,0], sqrt(sigma[0,0]))

pl.hist(xsamples, normed=True, bins=20, color='g')
pl.plot(xs, vs, 'r-')

pl.show()

# \mu = 1, \sigma = \sqrt{0.3}
# p_1(x) = \frac{1}{\sqrt{0.3*2\pi}} \exp \left\{ -\frac{1}{2 \cdot 0.3} (x - 1)^2 \right\}

## Question 1.7

f, axes = pl.subplots(2, 3)
f.suptitle('Question 1.7')

bins = [10,15,20]
ks, kxs, kys = gen_samples(1000)
for i in range(3):
    axes[0,i].set_title("bins = %dx%d" % (bins[i], bins[i]))
    axes[0,i].hist2d(kxs, kys, bins=(bins[i],bins[i]))

sample_nums = [100, 1000, 10000]
for i in range(3):
    s, xs, ys = gen_samples(sample_nums[i])
    axes[1,i].set_title("samples = %d" % sample_nums[i])
    axes[1,i].hist2d(xs, ys, bins=(15,15))

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

f, axes = pl.subplots(2)

l = 2.0
true_mean = 1.0/l
cats = range(10, 1000, 100)
sample_means = [
    [ fabs(true_mean - sum(exp_sample(l, n))/n) for i in range(1000) ]
    for n in cats
]

axes[0].boxplot(sample_means)
axes[0].set_xticks(range(1, len(cats)+1, 1), cats)
axes[1].boxplot(sample_means)
axes[1].set_xticks(range(1, len(cats)+1, 1), cats)
axes[1].set_yscale('log')

pl.show()


## Question 1.9

# Load image
img = Image.open('kande1.jpg')
(ysize, xsize) = img.size
im = img.load()

# Extract training set
training_set = [np.array(im[x,y]).reshape((3,1))*1.0 for x in range(150,330) for y in range(264,328)]

# Convert image to list of pixels
image = [np.array(im[y,x]).reshape((3,1))*1.0 for x in range(xsize) for y in range(ysize)]

# Maximum likelyhood solution for sample mean
mu_ml = sum(training_set) / len(training_set)

# Maximum likelyhood solution for covariance matrix
sigma_ml = sum([np.dot((x - mu_ml),(x - mu_ml).T) for x in training_set]) / len(training_set)

# Covariance matrix inverse
sigma_ml_inv = np.linalg.inv(sigma_ml)

# Probability density function
def prob_density(x, mu, sigma, sigma_inv):
    try:
        exponent = -(0.5)*(np.dot((x-mu).T,np.dot(sigma_inv, (x-mu))))
    except Exception as e:
        print x, mu, sigma, sigma_inv, e
    denominator = (2.0*np.pi)**(1.5) * np.linalg.norm(sigma)**(0.5)
    return 1/denominator * np.exp(exponent)

# Calculate probability density for the image
probabilities = np.array([prob_density(x, mu_ml, sigma_ml, sigma_ml_inv) for x in image]).reshape(xsize,ysize)

pitcher_img = 255*probabilities

# good color map: pl.cm.afmhot
COLORMAP = pl.cm.PuRd_r

pl.figure()
pl.title('Question 1.9')
pl.imshow(pitcher_img, interpolation='nearest', cmap=COLORMAP)
pl.show()

# Question 1.10

# Weighted average position
weighted_points = [np.array([x,y])*probabilities[x,y] for x in range(xsize) for y in range(ysize)]
Z = sum(probabilities.reshape(ysize*xsize))
q_hat = sum(weighted_points) / Z

# Spacial covariance
prob_sum = np.zeros((2,2))
for x in range(xsize):
    for y in range(ysize):
        q = np.array([y,x]).reshape(2,1)
        part = np.dot((q-q_hat),(q-q_hat).T) * probabilities[x,y]
        prob_sum += part

C = prob_sum / Z
C_inv = np.linalg.inv(C)

probabilities = np.array([prob_density(np.array([x,y]), q_hat, C, C_inv) for x in range(xsize) for y in range(ysize)]).reshape(xsize,ysize)

pl.figure()
pl.title('Question 1.10')
pl.imshow(pitcher_img, interpolation='nearest', cmap=COLORMAP)
pl.contour(probabilities)
pl.scatter(q_hat[1], q_hat[0], c='g', s=100, marker='v')
pl.show()


# Question 1.11
# Mostly copy&paste from 1.9 because of dependencies

# Load image
img = Image.open('kande2.jpg')
(ysize, xsize) = img.size
im = img.load()

# Convert image to list of pixels
image = [np.array(im[y,x]).reshape((3,1))*1.0 for x in range(xsize) for y in range(ysize)]

# Calculate probability density for the image
probabilities = np.array([prob_density(x, mu_ml, sigma_ml, sigma_ml_inv) for x in
    image]).reshape(xsize,ysize)

# Weighted average position
weighted_points = [np.array([x,y])*probabilities[x,y] for x in range(xsize) for y in range(ysize)]
weighted_average_pos = sum(weighted_points)  / sum(probabilities.reshape(ysize*xsize))

pl.figure()
pl.title('Question 1.11')
pl.imshow(255*probabilities, interpolation='nearest', cmap=pl.cm.afmhot)
pl.scatter(weighted_average_pos[1], weighted_average_pos[0], c='g', s=100, marker='v')
pl.show()
