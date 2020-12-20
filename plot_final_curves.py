import numpy as np
import matplotlib.pyplot as plt

nets = ['FB', 'BA', 'WS']
path = 'fig/FinalCurves/'

plt.figure(figsize=(7, 6))
plt.xlabel('r / (<k> + 1)')
plt.ylabel('Average median contribution')

for net in nets:
    x = np.load(path + 'x-' + net + '.npy')
    y = np.mean(np.load(path + 'y-' + net + '.npy'), axis=0)
    error_aux = np.load(path + 'error-' + net + '.npy')
    error = np.zeros(len(x))
    for i in range(error_aux.shape[0]):
        error += error_aux[i, :] ** 2
    error = np.sqrt(error / error_aux.shape[0])
    plt.errorbar(x, y, error, label=net, marker='o')

plt.legend()
plt.show()
