# ===========================================================================
# Conclusion
# * Higher downsample rate require more iteration of E-M algorithm
# * enable stochastic_downsampling will significant reduce the
# number of iteration
# ===========================================================================
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from odin.ml import GMM
from odin import visual as V

np.random.seed(1234)
nmix = 8
pdf_path = '/tmp/tmp.pdf'

# ===========================================================================
# Generate Artificial data
# ===========================================================================
X = []
y = []
stats_mean = []
stats_sigma = []
for i in range(nmix):
  m = (np.random.randint(-18, 18, size=(1, 2)) +
       np.random.randint(-18, 18, size=(1, 2)))
  s = np.random.rand(1, 2) + np.random.rand(1, 2)
  stats_mean.append(m)
  stats_sigma.append(np.diag(s.ravel()))
  dat = m + s * np.random.randn(512, 2)
  X.append(dat)
  y.append(dat)
X = np.concatenate(X, axis=0)
print(X.shape)

stats_mean = np.concatenate(stats_mean, axis=0)

# ===========================================================================
# Plot
# ===========================================================================
for niter in (8, 16, 128):
  for downsample in (1, 4, 16):
    for stochastic in (True, False):
      gmm = GMM(nmix=nmix, nmix_start=1, niter=niter,
                allow_rollback=True, exit_on_error=True,
                downsample=downsample,
                stochastic_downsample=stochastic,
                batch_size_cpu=25,
                batch_size_gpu=25,
                device='gpu')
      gmm.initialize(X)
      print(gmm)
      gmm.fit(X)
      # ====== match each components to closest mean ====== #
      gmm_mean = [None] * nmix
      gmm_sigma = [None] * nmix
      for mean, sigma in zip(gmm.mean.T, gmm.sigma.T):
        sigma = np.diag(sigma)
        distance = sorted([(i, np.sqrt(np.sum((m - mean)**2)))
                           for i, m in enumerate(stats_mean)],
                          key=lambda x: x[1])
        for i, dist in distance:
          if gmm_mean[i] is None:
            gmm_mean[i] = mean
            gmm_sigma[i] = sigma
            break
      # ====== plot everything ====== #
      plt.figure()
      colors = V.generate_random_colors(n=nmix)
      for i in range(nmix):
        c = colors[i]
        dat = y[i]
        sigma = gmm_sigma[i]
        plt.scatter(dat[:, 0], dat[:, 1], c=c, s=0.5)
        V.plot_ellipses(gmm_mean[i], gmm_sigma[i], alpha=0.5, color=c)
        V.plot_ellipses(stats_mean[i], stats_sigma[i], alpha=0.3, color='red')
      plt.suptitle('#iter:%d stochastic:%s downsample:%d ' %
        (niter, stochastic, downsample))
V.plot_save(pdf_path)
