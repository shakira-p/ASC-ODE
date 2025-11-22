import numpy as np
import matplotlib.pyplot as plt

type = ["explicit", "implicit", "improved", "crank"]
step = "200"
data = []

colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

for t in type:
    data.append(np.loadtxt(f"data/{t}{step}.txt", usecols=(0, 1, 2)))

for i in range(len(data)):
    plt.plot(data[i][:,0], data[i][:,1], '--', label=f'{type[i]} position', color=colors[i])
    plt.plot(data[i][:,0], data[i][:,2], label=f'{type[i]} velocity', color=colors[i])
plt.xlabel('time')
plt.ylabel('value')
plt.title('Mass-Spring System Time Evolution')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.savefig(f"./plots/MassSpringSystemTimeEvolution_{step}.png")
plt.show()


for i in range(len(data)):
    if i == 2:
        plt.plot(data[i][:,1], data[i][:,2],label=f'{type[i]}', color=colors[i], linewidth=3.0)
    else:
        plt.plot(data[i][:,1], data[i][:,2], label=f'{type[i]}', color=colors[i])
plt.xlabel('position')
plt.ylabel('velocity')
plt.title('Mass-Spring System Phase Plot')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid()
plt.savefig(f"./plots/MassSpringPhasePlot_{step}.png")
plt.show()

