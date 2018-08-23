
# ====== Visual cluster ====== #
# TODO: fix bug of the scatter method
labels = list(set(filter(lambda x: len(x) == 1,
                         [i.split('_')[-1] for i in all_files])))
print("Labels:", ctext(labels, 'cyan'))
for feat in ('bnf', 'mspec', 'spec', 'mfcc'):
  if feat not in ds:
    continue
  from sklearn.manifold import TSNE
  X = []; y = []
  # get right feat and indices
  feat_pca = ds.find_prefix(feat, 'pca')
  indices = ds.find_prefix(feat, 'indices')
  # transform
  prog = Progbar(target=len(indices),
                 print_summary=True, print_report=True,
                 name="PCA transform: %s" % feat)
  for f, (start, end) in indices:
    if len(f.split('_')[-1]) == 1:
      X.append(np.mean(
          feat_pca.transform(ds[feat][start:end]),
          axis=0, keepdims=True))
      y.append(f.split('_')[-1])
    prog.add(1)
  X_pca = np.concatenate(X, axis=0)
  y = np.asarray(y)
  with UnitTimer(name="TSNE: feat='%s' N=%d" % (feat, X_pca.shape[0])):
    X_tsne = TSNE(n_components=2).fit_transform(X_pca)
  colors = V.generate_random_colors(len(labels), seed=12082518)
  # conver y to appropriate color
  y = [colors[labels.index(i)] for i in y]
  legend = {c: str(i) for i, c in enumerate(colors)}
  with V.figure(ncol=1, nrow=5, title='PCA: %s' % feat):
    V.plot_scatter(X_pca[:, 0], X_pca[:, 1], color=y, legend=legend)
  with V.figure(ncol=1, nrow=5, title='TSNE: %s' % feat):
    V.plot_scatter(X_tsne[:, 0], X_tsne[:, 1], color=y, legend=legend)
# ====== save all the figure ====== #
V.plot_save(os.path.join(fig_path, 'pca_tsne.pdf'),
            tight_plot=True)
# ====== print log ====== #
print('Output path:', ctext(output_path, 'cyan'))
print('Figure path:', ctext(fig_path, 'cyan'))
print('Log path:', ctext(log_path, 'cyan'))
