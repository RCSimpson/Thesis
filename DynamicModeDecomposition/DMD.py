import numpy as np
from numpy import matlib
import warnings

########################################################################################################################
# DMD Methods and associated Functions
########################################################################################################################

def standard_dynamic_mode_decomposition(Gtot, thrshhld):
    mrow, ncol = Gtot.shape
    Ga = np.matlib.repmat((np.mean(Gtot, 1)).reshape(mrow, 1), 1, ncol)
    gfluc = Gtot - Ga
    gm = gfluc[:, :ncol - 1]
    gp = gfluc[:, 1:]

    u, s, vh = np.linalg.svd(gm, full_matrices=False)
    sm = np.max(s)
    indskp = np.log10(s / sm) > -thrshhld
    sr = s[indskp]
    ur = u[:, indskp]
    v = np.conj(vh.T)
    vr = v[:, indskp]
    kmat = gp @ vr @ np.diag(1. / sr) @ np.conj(ur.T)
    evls, evcs = np.linalg.eig(kmat)
    phim = (np.linalg.solve(evcs, gm)).T
    return evcs, evls, phim, Ga

def polynomial_kernel():
    pass

def sinc_kernel():
    pass

def radial_basis_functions_kernel():
    pass

def kernel_evluation():
    pass

def kernel_dynamic_mode_decomposition(data, number_of_terms, threshhold):
    rows, columns = data.shape
    data_average = np.matlib.repmat((np.mean(data), 1).reshape(rows, 1), 1, columns)
    data_fluctuations_about_average = data - data_average
    data_plus = data_fluctuations_about_average[:, 1:]
    data_minus = data_fluctuations_about_average[:, :-1]
    extended_data = np.zeros((rows-1,columns-1), dtype=np.float64)
    a_matrix = np.zeros((rows - 1, columns - 1), dtype=np.float64)
    kernel_evluation(data_minus, data_plus, rows, columns-1, number_of_terms, a_matrix, extended_data)
    q, ssq, qh = np.linalg.svd(data, full_matrices=False)

########################################################################################################################
# Error Computation
########################################################################################################################