import smn_mod as smn
import numpy as np

lines = np.array((0,1,2,3,4,5,4,3,2,1), dtype=np.int64)
columns = np.array((1,2,3,4,5,4,3,2,1,0), dtype=np.int64)
values = np.array((1,123,12,2,2000,1,33,45,78,1), dtype=np.float64)

A = B = C = smn.sparse_matrix(lines, columns, values)
A.to_dense()
lowest_val = np.float64(B.regularize(2, 124))
print("Lowest val in regularized B: " + str(lowest_val))
B.to_dense()
C.renormalize(3, 120)
print("Lowest val in regularized B: " + str(lowest_val))
C.to_dense()
quit()

