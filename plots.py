import os,sys


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
})

desired = np.loadtxt("data/trajectory_curve_20Hz.csv", delimiter=",")

done = np.loadtxt("state_trajectory.csv", delimiter=",")
times = np.loadtxt("time_step.csv", delimiter=",")

plt.plot(done[:,0],done[:,1])
plt.plot(desired[:,0],desired[:,1])
plt.show()

plt.plot(np.arange(len(done)),done[:,3])
plt.plot(np.arange(len(done)),done[:,4])
plt.show()

plt.plot(np.arange(len(times)),times)
plt.show()



