import pandas as pd

from odin import visual as vs
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

# === 1. varying zdim and the active units MNIST
# zdim = [2, 5, 10, 20, 35, 60, 80]
# au = [2, 5, 7, 7, 7, 7, 7]
# llk = [-134.05, -103.22, -97.78, -98.35, -99.04, -96.23, -96.27]
# df = pd.DataFrame({'zdim': zdim, 'active latent units': au, 'llk': llk})
# print(df)
#
# plt.figure(figsize=(5, 4), dpi=200)
# sns.scatterplot(x='zdim', y='llk', data=df,
#                 hue='active latent units',
#                 size='active latent units',
#                 sizes=(80, 150),
#                 alpha=0.9)
# plt.xticks(zdim, [str(i) for i in zdim])
# vs.plot_save(verbose=True)


# === 2. varying zdim and the active units CIFAR10
# zdim = [32, 64, 128, 256, 512, 512, 1024]
# au = [32, 64, 128, 256, 466, 512, 466]
# llk = [-13605.55, -12715.09, -11767.64, -10701.66, -9662.88, -142.95,
#        -9653.24]
# df = pd.DataFrame({'zdim': zdim, 'active latent units': au, 'llk': llk})
# print(df)
#
# plt.figure(figsize=(5, 4), dpi=200)
# sns.scatterplot(x='zdim', y='llk', data=df,
#                 hue='active latent units',
#                 size='active latent units',
#                 sizes=(80, 200),
#                 alpha=0.8)
# plt.gca().set_xscale('log')
# plt.xticks(zdim, [str(i) for i in zdim])
# vs.plot_save(verbose=True)

# === 3. varying py semafoVAE

py = [0.002, 0.004, 0.01, 0.05, 0.1, 0.2, 0.5]
llk = [-3456.38, -3460.43, -3457.71, -3456.63, -3457.03, -3456.75, -3456.91]
fid = [28.78, 27.80,  32.11, 26.84, 28.61, 28.43,  25.12]
dci = [60.84, 68.49, 74.22, 81.79, 80.88, 83.72, 85.12]

plt.figure(figsize=(10, 3), dpi=200)

plt.subplot(1, 3, 1)
plt.plot(py, llk, label='SemafoVAE')
plt.plot([py[0], py[-1]], [-3464.40, -3464.40], label='VAE baseline', color='r')
plt.gca().set_xscale('log')
plt.xticks(py, [str(i) for i in py], rotation=-30)
plt.legend(fontsize=8)
plt.xlabel('Supervision rate')
plt.title('Test log-likelihood')

plt.subplot(1, 3, 2)
plt.plot(py, fid, label='SemafoVAE')
plt.plot([py[0], py[-1]], [74.57, 74.57], label='VAE baseline', color='r')
plt.gca().set_xscale('log')
plt.xticks(py, [str(i) for i in py], rotation=-30)
plt.legend(fontsize=8)
plt.xlabel('Supervision rate')
plt.title('FID')

plt.subplot(1, 3, 3)
plt.plot(py, dci, label='SemafoVAE')
plt.plot([py[0], py[-1]], [64.82, 64.82], label='VAE baseline', color='r')
plt.gca().set_xscale('log')
plt.xticks(py, [str(i) for i in py], rotation=-30)
plt.legend(fontsize=8)
plt.xlabel('Supervision rate')
plt.title('DCI')

plt.tight_layout()
vs.plot_save(verbose=True)
