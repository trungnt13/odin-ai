import numpy as np
from scipy.optimize import linear_sum_assignment


def search_assignment(matrix,
                      row_assignment=False,
                      maximize=True,
                      inplace=False):
  r"""Solve the linear sum assignment problem.

  This function can also solve a generalization of the classic assignment
  problem where the cost matrix is rectangular. If it has more rows than
  columns, then not every row needs to be assigned to a column, and vice
  versa.

  Arguments:
    matrix : an Array.
      The cost matrix of the bipartite graph.
    inplace : a Boolean.
      If True, return a new matrix with the applied assignment

  Returns:
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.
  """
  assert matrix.ndim == 2
  rows, cols = linear_sum_assignment(matrix if row_assignment else matrix.T,
                                     maximize=maximize)
  # select the right assignment and keep the order of other dimension intact
  if not row_assignment:
    rows, cols = (cols, rows)
    ids = np.argsort(rows)
  else:
    ids = np.argsort(cols)
  rows = rows[ids]
  cols = cols[ids]
  # inplace output matrix
  if inplace:
    # make sure all indices appear
    rows = rows.tolist()
    for i in range(matrix.shape[0]):
      if i not in rows:
        rows.append(i)
    cols = cols.tolist()
    for i in range(matrix.shape[0]):
      if i not in cols:
        cols.append(i)
    return matrix[rows] if row_assignment else matrix[:, cols]
  return rows, cols
