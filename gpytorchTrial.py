#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import math
import matplotlib.pyplot as plt
import gpytorch
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import time


# In[2]:


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd


# In[3]:


start = time.time()
if (torch.cuda.is_available()):
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


# In[4]:


fineSpringConstants = pd.read_csv("fineSpringParameters.csv",header=None)


# In[5]:


timesteps = pd.read_csv("timesteps.csv",header=None)


# In[6]:


coarseSpringConstants = np.arange(0.1,51.1,5)
mediumSpringConstants = np.arange(0.1,51.1,1)


# In[7]:


fineDisplacements = pd.read_csv("fineDisplacements.csv",header=None)


# In[8]:


fineDisplacements.set_index(fineSpringConstants.to_numpy().squeeze(),inplace=True)


# In[9]:


fineDisplacements.columns = timesteps.to_numpy().squeeze()


# In[10]:


coarseDisplacements = fineDisplacements.loc[coarseSpringConstants]


# In[11]:


mediumDisplacements = fineDisplacements.loc[mediumSpringConstants]


# In[12]:


inputFeatures = np.meshgrid(timesteps, coarseSpringConstants)


# In[13]:


xt = np.stack((inputFeatures[1],inputFeatures[0]),axis=-1).reshape((-1,2))


# In[14]:


train_x = torch.from_numpy(xt)
train_x.to(device)


# In[15]:


yt = coarseDisplacements.to_numpy().flatten()


# In[16]:


train_y = torch.from_numpy(yt)
train_y.to(device)


# In[17]:


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
#likelihood.to(device)
model = ExactGPModel(train_x, train_y, likelihood)
#model.to(device)


# In[18]:


# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()


# In[19]:


testFeatures = np.meshgrid(timesteps,mediumSpringConstants)


# In[20]:


test_x = torch.from_numpy(np.stack((testFeatures[1],testFeatures[0]),axis=-1).reshape((-1,2)))
test_x.to(device)


# In[21]:


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(): #gpytorch.settings.fast_pred_var():
    #test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))


# In[22]:


end = time.time()
print("Time elapsed",start-end)


# In[49]:


with torch.no_grad():
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #fig = plt.figure()
    #ax = Axes3D(fig)
    surf = ax.plot_trisurf(test_x[0:,0].numpy(), test_x[0:,1].numpy(),observed_pred.mean.numpy(), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    #plt.imshow(z)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.scatter(train_x[0:,0].numpy(),train_x[0:,1].numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    #ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #ax.set_ylim([-3, 3])
    ax.legend([ 'Mean','Observed Data'])
    plt.show()
    


# In[48]:





# In[42]:





# In[ ]:




