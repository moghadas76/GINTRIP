import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data (you'll need to replace this with your actual data)
sparsity = np.load("test_loader_sparsity.npy")
fidels = np.load("test_loader_fidelity_plus_list.npy")*100
# sparsity = np.linspace(0.0, 1.0, 10)
fidelity1 = np.array([0]*50)
fidelity2 = np.array([0]*50)
fidelity3 = np.array([0]*50)
fidelity4 = np.array([0]*50)
# fidelity5 = np.array([0]*50)
fidelity1 = np.where(fidelity1 == 0, np.nan, fidelity1)
fidelity2 = np.where(fidelity2 == 0, np.nan, fidelity2)
fidelity3 = np.where(fidelity3 == 0, np.nan, fidelity3)
fidelity4 = np.where(fidelity4 == 0, np.nan, fidelity4)
fidelity1[50 -3], fidelity1[50 -4], fidelity1[50 -14], fidelity1[50 -24], fidelity1[50 -34], fidelity1[50 -39], fidelity1[50 -44]  = 0.01, 0.015, 0.014, 0.011, 0.008, 0.006, 0.001
fidelity2[50 -11], fidelity2[50 -17], fidelity2[50 -22], fidelity2[50 -27], fidelity2[50 -33], fidelity2[50 -42], fidelity2[50 -45], fidelity2[50-47], fidelity2[50-48] = 0.01, 0.01, 0.014, 0.015, 0.016, 0.017, 0.018, 0.016, 0.015
fidelity3[0], fidelity3[2], fidelity3[6], fidelity3[9], fidelity3[13], fidelity3[18], fidelity3[22]\
, fidelity3[50-28], fidelity3[50-34], fidelity3[50-41]  = 0.23, 0.21, 0.195, 0.18, 0.15, 0.13, 0.105, 0.06, 0.055, 0.045
fidelity4[50-7], fidelity4[50-11], fidelity4[50-15], fidelity4[50-20], fidelity4[50-25], fidelity4[50-30], fidelity4[50-36], fidelity4[50-38], \
     fidelity4[50-45], fidelity4[50-46] = 0.20, 0.15, 0.13, 0.12, 0.11, 0.09, 0.065, 0.045, 0.025, 0.015
valid_indices = ~np.isnan(fidelity1)
valid_indices2 = ~np.isnan(fidelity2)
valid_indices3 = ~np.isnan(fidelity3)
valid_indices4 = ~np.isnan(fidelity4)
# fidelity1[3] = 0.01, 0.02, 0.03, 0.05, 0.08, 0.11, 0.15, 0.17, 0.19, 0.21
# fidelity1[4] = 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.17
#  = 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
# 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015
# Create the plot
plt.figure(figsize=(10, 8))

# Plot the lines
plt.plot(sparsity[valid_indices], fidelity1[valid_indices], color='#66c2a5', marker='s', markersize=8, linewidth=2, label='GNNExplainer[9]')
plt.plot(sparsity[valid_indices2], fidelity2[valid_indices2], color='#fc8d62', marker='x', markersize=8, linestyle='-.', linewidth=2, label='GraphMask[5]')
# plt.plot(sparsity[valid_indices3], fidelity3[valid_indices3], color='#8da0cb', marker='*', markersize=10, linestyle='--', linewidth=2, label='PGExplainer')
plt.plot(sparsity[valid_indices4], fidelity4[valid_indices4], color='#e78ac3', marker='o', markersize=8, linewidth=2, label='STExplainer[14]')
plt.plot(sparsity, fidels, color='#a6d854', marker='v', markersize=8, linestyle=':', linewidth=2, label='Ours')

# Customize the plot
# plt.title('Spatial Graph', fontsize=16)
plt.xlabel('Sparsity($k$)', fontsize=14)
plt.ylabel('Fidelity(+)', fontsize=14)
plt.xlim(0.0, 1.0)
plt.ylim(0, 0.25)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Customize ticks
plt.xticks(np.arange(1.0, -0.1, -0.2))
plt.yticks(np.arange(0, 0.26, 0.05))

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

