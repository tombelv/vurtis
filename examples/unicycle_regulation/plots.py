import numpy as np
import matplotlib.pyplot as plt


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 12,
# })

executed = np.loadtxt("data/state_trajectory.csv", delimiter=",")
control = np.loadtxt("data/input_trajectory.csv", delimiter=",")

executed_p = np.loadtxt("data/state_trajectory_p.csv", delimiter=",")
control_p = np.loadtxt("data/input_trajectory_p.csv", delimiter=",")

sensitivity = np.loadtxt("data/sensitivity.csv", delimiter=",")
#times = np.loadtxt("data/time_step.csv", delimiter=",")

print((executed_p[0, 0:-1:3]-executed[0, 0:-1:3])/1e-6 - sensitivity[4, 0:-1:3])
print((executed_p[0, 1:-1:3]-executed[0, 1:-1:3])/1e-6 - sensitivity[4, 1:-1:3])
print((executed_p[0, 2:-1:3]-executed[0, 2:-1:3])/1e-6 - sensitivity[4, 2:-1:3])


plt.plot(executed[:, 0], executed[:, 1])
plt.show()


plt.plot((executed_p[0, 2:-1:3]-executed[0, 2:-1:3])/(1e-6))
plt.plot(sensitivity[4, 2:-1:3])
plt.show()

plt.plot(np.arange(len(control)), control[:, 0])
plt.plot(np.arange(len(control)), control[:, 1])
plt.show()

#plt.plot(np.arange(len(times)), times)
#plt.show()



