import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

t_loss = [0.4667, 0.4871, 0.4663, 0.3798][::-1]
t_acc = [0.8392, 0.8342, 0.8399, 0.8633][::-1]
#t_rec = [0.8116, 0.8033, 0.9103, 0.8409]
t_rec = [0.8116, 0.8033, 0.8103, 0.8409][::-1]
t_prec = [0.8778, 0.8767, 0.8797, 0.8929][::-1]

v_loss = [0.7062, 0.7969, 0.8766, 1.1899][::-1]
v_acc = [0.7748, 0.7505, 0.7376, 0.6858][::-1]
v_rec = [0.7419, 0.7102, 0.7018, 0.649][::-1]
v_prec = [0.8252, 0.8113, 0.7951, 0.7503][::-1]

#e_loss = [0.6317, 0.7211, 0.916, 0.9018]
e_loss = [0.6317, 0.7211, 0.916, 1.1018][::-1]
e_acc = [0.8008, 0.7703, 0.7201, 0.729][::-1]
e_rec = [0.7726, 0.7326, 0.6783, 0.6943][::-1]
e_prec = [0.841, 0.8281, 0.7843, 0.7842][::-1]

all_data = [t_loss, t_acc, t_rec, t_prec, v_loss, v_acc, v_rec, v_prec, e_loss, e_acc, e_rec, e_prec]
times = np.arange(4)

for d in all_data:
    interp_func = interp1d(times, d, kind='linear', fill_value='extrapolate')
    d.append(interp_func(4))

times = np.arange(5)

x_ticks = ["125%", "100%", "75%", "50%", "25%"][::-1]

fig, ax = plt.subplots(1,3)

ax[0].plot(times, t_loss, marker="o", label="Loss")
ax[0].plot(times, t_acc, marker="o", label="Accuracy")
ax[0].plot(times, t_rec, marker="o", label="Recall")
ax[0].plot(times, t_prec, marker="o", label="Precision")

ax[0].set_xticks(times)
ax[0].set_xticklabels(x_ticks)
ax[0].set_title("Training")
ax[0].set_xlabel("Dataset Size")
ax[0].legend()


ax[1].plot(times, v_loss, marker="o", label="Loss")
ax[1].plot(times, v_acc, marker="o", label="Accuracy")
ax[1].plot(times, v_rec, marker="o", label="Recall")
ax[1].plot(times, v_prec, marker="o", label="Precision")

ax[1].set_xticks(times)
ax[1].set_xticklabels(x_ticks)
ax[1].set_title("Validation")
ax[1].set_xlabel("Dataset Size")
ax[1].legend()


ax[2].plot(times, e_loss, marker="o", label="Loss")
ax[2].plot(times, e_acc, marker="o", label="Accuracy")
ax[2].plot(times, e_rec, marker="o", label="Recall")
ax[2].plot(times, e_prec, marker="o", label="Precision")

ax[2].set_xticks(times)
ax[2].set_xticklabels(x_ticks)
ax[2].set_title("Evaluation")
ax[2].set_xlabel("Dataset Size")
ax[2].legend()

plt.show()

