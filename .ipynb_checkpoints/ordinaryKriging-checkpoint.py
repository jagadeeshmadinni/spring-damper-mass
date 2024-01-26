import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pandas as pd
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
%matplotlib inline
S = pd.read_csv("springParameters.csv",header=None).to_numpy()
K = S[0:20]
t = pd.read_csv("timesteps.csv",header=None).to_numpy()
Z = pd.read_csv("displacement.csv",header=None).transpose().to_numpy()
Z = Z[0:20,:]
X,Y = np.meshgrid(t,K[0][:20])
Z = np.reshape(Z,601*20)
X = np.reshape(X,601*20)
Y = np.reshape(Y,601*20)
gridx = np.arange(0.0, 61,0.1)
gridy = np.arange(1,20,0.2)
OK = OrdinaryKriging(
    X,
    Y,
    Z,
    variogram_model="linear",
    verbose=True,
    enable_plotting=True,
)
z, ss = OK.execute("grid", gridx, gridy)
gridX,gridY = np.meshgrid(gridx,gridy) 
#kt.write_asc_grid(gridx, gridy, z, filename="output.asc")

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(gridX, gridY, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
#plt.imshow(z)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()