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

########################################################################################################################
# Error Computation
########################################################################################################################