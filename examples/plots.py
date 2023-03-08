import os,sys


import numpy as np
import matplotlib.pyplot as plt


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 12,
# })

desired = np.loadtxt("data/trajectory_curve_20Hz.csv", delimiter=",")

executed = np.loadtxt("state_trajectory.csv", delimiter=",")
times = np.loadtxt("time_step.csv", delimiter=",")

plt.plot(executed[:,0],executed[:,1])
plt.plot(desired[:,0],desired[:,1])
plt.show()

plt.plot(np.arange(len(executed)),executed[:,3])
plt.plot(np.arange(len(executed)),executed[:,4])
plt.show()

plt.plot(np.arange(len(times)),times)
plt.show()



