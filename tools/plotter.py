import numpy as np
import matplotlib.pyplot as plt

f = open('log_qwe_lambda20_power4_newnn.dat', 'r')
energies = []
for line in f:
	if 'overlap' not in line:
		continue
	energy = float(line.split()[-3])
	overlap = float(line.split()[-1])

	energies.append(energy)
f.close()

energies = np.sort(np.unique(energies))


overlaps = np.zeros(shape = (len(energies), 1000))
f = open('log_qwe_lambda20_power4_newnn.dat', 'r')
n_epoch = -1
for line in f:
	if '#' in line:
		n_epoch += 1
	if 'overlap' not in line:
		continue
	energy = float(line.split()[-3])
	overlap = float(line.split()[-1])
	overlaps[np.where(energies == energy)[0][0], n_epoch] = overlap
f.close()

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')


max_epoch = 322

for i in range(max_epoch):
	print(np.sum(overlaps[:, i] ** 2))

for energy in energies:
	plt.plot(np.arange(0, max_epoch), overlaps[np.where(energy == energies)[0][0], :max_epoch] ** 2, label = '$E = ' + str(energy) + '$')

plt.ylabel('overlap squared', fontsize = 16)
plt.xlabel('$n$ epoch', fontsize = 16)
plt.yscale('log')
plt.grid(True)
plt.legend(loc = 'upper left', ncol = 2, fontsize = 8)

plt.show()