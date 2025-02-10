import numpy as np
import matplotlib.pyplot as plt

out_dir = "ibm_deltas_u0/"
param_label = "u0"
param_values = [1, 2, 4, 8]

out_dir = ""
param_label = "dt"
param_values = [2, 1, 0.5, 0.25]

delta_vn = np.genfromtxt(out_dir + "ibm_delta_vn.csv", delimiter=",")[:, :-1]
delta_w  = np.genfromtxt(out_dir + "ibm_delta_w.csv", delimiter=",")[:, :-1]

fig = plt.figure(1); plt.clf(); plt.show(block=False)
(ax1, ax2) = fig.subplots(2, 1, sharex=True)

def plot_min_max(ax, data):
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    markers = ["P", "C"]
    for i, row in enumerate(data):
        min_values = row[::2]
        max_values = row[1::2]
        color = colors[i % len(colors)]
        label = f"{param_label} = {param_values[i]}"
        ax.plot(max_values, color=color, label=label)
        ax.plot(min_values, color=color, linestyle="--")
        for j, (x, y) in enumerate(zip(range(len(max_values)), max_values)):
            ax.annotate(markers[j % len(markers)], (x, y), textcoords="offset points", xytext=(0,0), ha="center")

# Plot delta_vn on the top subplot
plot_min_max(ax1, delta_vn)
ax1.set_ylabel("Delta vn")
ax1.legend()

# Plot delta_w on the bottom subplot
plot_min_max(ax2, delta_w)
ax2.set_ylabel("Delta w")
ax2.legend()

plt.tight_layout()
plt.draw()