import numpy as np


def subgraph_count(Af):
    fvec = np.zeros(14)
    nds, nds = Af.shape
    dgs = np.zeros((nds, 1), dtype=np.int)
    ovec = np.ones((nds, 1), dtype=np.int)
    dfvec = np.zeros((nds, 1), dtype=np.int)
    dftvec = np.zeros((nds, 1), dtype=np.int)
    dgs[:, 0] = np.sum(Af, 1)
    dfvec[:, 0] = dgs[:, 0] - ovec[:, 0]
    dftvec[:, 0] = dgs[:, 0] - 2 * ovec[:, 0]

    A2 = Af @ Af
    A2f = Af * A2
    # Q = np.multiply(A2,A)
    A3 = Af @ A2
    A3d = np.diag(A3)
    A4 = Af @ A3
    A5 = Af @ A4
    P1 = np.sum(dgs) / 2
    P2 = np.sum(dgs * dfvec) / 2.
    A2m1 = A2 - np.ones(A2.shape)
    a2chs = .5 * A2 * A2m1
    a2chsvec = np.sum(a2chs - np.diag(np.diag(a2chs)), 1)
    # note, first three entries of fvec are the number of cycles C_3, C_4, and C_5

    fvec[0] = np.trace(A3) / 6.  # n_G(C_3)
    fvec[1] = (np.trace(A4) - 4. * P2 - 2. * P1) / 8.  # n_G(C_4)
    fvec[3] = .5 * np.sum(np.sum(Af * (dfvec @ dfvec.T))) - 3. * fvec[0]  # n_G(H_3)
    fvec[4] = np.sum(dgs[:, 0] * dfvec[:, 0] * dftvec[:, 0]) / 6.  # n_G(H_4)
    fvec[5] = np.sum(A3d * dftvec[:, 0]) / 2.  # n_G(H_5)
    fvec[2] = (np.trace(A5) - 10 * fvec[5] - 30 * fvec[0]) / 10.  # n_G(C_5)

    fvec[6] = 1 / 4. * np.sum(np.sum(A2f * A2m1))  # n_G(H_6)
    fvec[7] = 1 / 4. * np.dot(A3d, (dftvec[:, 0] * (dftvec[:, 0] - np.ones(nds))))  # n_G(H_7)
    fvec[8] = 1 / 2. * (dftvec.T @ (A2f @ dftvec)) - 2. * fvec[6]  # n_G(H_8)
    fvec[9] = np.dot(dftvec[:, 0], a2chsvec) - 2. * fvec[6]  # n_G(H_9)
    fvec[10] = .5 * np.dot(A3d, np.sum(A2 - np.diag(np.diag(A2)), 1)) - 6. * fvec[0] - 2. * fvec[5] - 4. * fvec[
        6]  # n_G(H_10)
    fvec[11] = .5 * np.sum(.5 * A3d * (.5 * A3d - 1.)) - 2. * fvec[6]  # n_G(H_11)
    fvec[12] = .5 * np.sum(np.sum(A2f * A3)) - 9. * fvec[0] - 2. * fvec[5] - 4. * fvec[6]  # n_G(H_12)
    fvec[13] = 1. / 12. * np.sum(np.sum(A2f * (A2 - np.ones(A2.shape)) * (A2 - 2. * np.ones(A2.shape))))  # n_G(H_13)
    return fvec

