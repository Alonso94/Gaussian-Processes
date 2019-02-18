import gpflow
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# plot the model
def plot(m,x,y):
    xx=np.linspace(-0.1,1.1,100).reshape(100,1)
    mean,var=m.predict_y(xx)
    plt.figure(figsize=(12,6))
    plt.plot(x,y,'kx',mew=2)
    plt.plot(xx,mean,'C0',lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0]-2*np.sqrt(var[:,0]),
                     mean[:,0]+2*np.sqrt(var[:,0]),
                     color='C0',alpha=0.2)
    plt.xlim(-0.1,1.1)
    plt.show()

# data
n=24
x=np.random.rand(n,1)
y=np.sin(12*x)+0.66*np.cos(25*x)+np.random.randn(n,1)*0.1+3
plt.plot(x,y,'kx',mew=2)
plt.show()

# model
k=gpflow.kernels.Matern52(1,lengthscales=0.3)
m=gpflow.models.GPR(x,y,kern=k)
m.likelihood.variance=0.01
plot(m,x,y)

# ML estimate of parameters
gpflow.train.ScipyOptimizer().minimize(m)
plot(m,x,y)

# obtain posterior over hyperparameters in GP regression
m.clear()
m.kern.lengthscales.prior=gpflow.priors.Gamma(1.0,1.0)
m.kern.variance.prior=gpflow.priors.Gamma(1.0,1.0)
m.likelihood.variance.prior=gpflow.priors.Gamma(1.0,1.0)
m.compile()

sampler=gpflow.train.HMC()
samples=sampler.sample(m,num_samples=gpflow.test_util.notebook_niter(500),epsilon=0.05,lmin=10,lmax=20,logprobs=False)

def plot_samples(samples,x,y):
    xx=np.linspace(-0.1,1.1,100)[:,None]
    plt.figure(figsize=(12,6))
    for i,s in samples.iloc[::20].iterrows():
        f=m.predict_f_samples(xx,1,initialize=False,feed_dict=m.sample_feed_dict(s))
        plt.plot(xx,f[0,:,:],'C0',lw=2,alpha=0.1)
    plt.plot(x,y,'kx',mew=2)
    _=plt.xlim(xx.min(),xx.max())
    _=plt.ylim(0,6)
    plt.show()

plot_samples(samples,x,y)
