import numpy as np
import matplotlib.pyplot as plt

nets = ['FB', 'BA', 'WS']
path = 'fig/FinalCurves/'

plt.figure(figsize=(7, 6))
plt.xlabel('r / (<k> + 1)')
plt.ylabel('Average median contribution')

for net in nets:
    x = np.load(path + 'x-' + net + '.npy')
    list_y = np.load(path + 'y-' + net + '.npy')
    y = np.mean(list_y, axis=0)
    list_error = np.load(path + 'error-' + net + '.npy')
    error_aux = np.std(list_y, axis=0) / np.sqrt(list_y.shape[0] - 1)
    error_aux = error_aux**2
    error = np.zeros(len(x))
    for i in range(list_error.shape[0]):
        error += list_error[i, :] ** 2
    error = error / list_error.shape[0]**2
    error += error_aux
    error = np.sqrt(error)
    plt.errorbar(x, y, error, label=net, marker='o', markersize=3, linestyle=':')

plt.legend()
plt.show()
