import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# ======================================================================================================================
# ======================================================================================================================
# Twitter Simulation Model
# ======================================================================================================================
# ======================================================================================================================

# The Twitter Model has three possible events
# T1 add a new message node.
# T2 add a new retweet node and an edge between that node and an existing node via preferential attachment scheme
# T2 is essentially already covered in the pref model (motif_evolution.py) so this script focuses on T3.
# T3 add an edge between two existing nodes

# ======================================================================================================================
# Motif Counts
# ======================================================================================================================

def subgraph_counter(G, from_graph=False):
    if from_graph:
        Af = nx.adjacency_matrix(G)
    else:
        Af=G

    fvec = np.zeros(14, dtype=int)
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

    associated_strings = ['C3','C4','C5','H3','H4','H5','H6','H7','H8','H9','H10','H11','H12', 'H13']
    fvec[0] = np.trace(A3) / 6.  # n_G(C_3)
    fvec[1] = (np.trace(A4) - 4. * P2 - 2. * P1) / 8.  # n_G(C_4)
    fvec[2] = (np.trace(A5) - 10 * fvec[5] - 30 * fvec[0]) / 10.  # n_G(C_5)
    fvec[3] = .5 * np.sum(np.sum(Af * (dfvec @ dfvec.T))) - 3. * fvec[0]  # n_G(H_3)
    fvec[4] = np.sum(dgs[:, 0] * dfvec[:, 0] * dftvec[:, 0]) / 6.  # n_G(H_4)
    fvec[5] = np.sum(A3d * dftvec[:, 0]) / 2.  # n_G(H_5)
    fvec[6] = 1 / 4. * np.sum(np.sum(A2f * A2m1))  # n_G(H_6)
    fvec[7] = 1 / 4. * np.dot(A3d, (dftvec[:, 0] * (dftvec[:, 0] - np.ones(nds))))  # n_G(H_7)
    fvec[8] = 1 / 2. * (dftvec.T @ (A2f @ dftvec)) - 2. * fvec[6]  # n_G(H_8)
    fvec[9] = np.dot(dftvec[:, 0], a2chsvec) - 2. * fvec[6]  # n_G(H_9)
    fvec[10] = .5 * np.dot(A3d, np.sum(A2 - np.diag(np.diag(A2)), 1)) - 6. * fvec[0] - 2. * fvec[5] - 4. * fvec[6]  # n_G(H_10)
    fvec[11] = .5 * np.sum(.5 * A3d * (.5 * A3d - 1.)) - 2. * fvec[6]  # n_G(H_11)
    fvec[12] = .5 * np.sum(np.sum(A2f * A3)) - 9. * fvec[0] - 2. * fvec[5] - 4. * fvec[6]  # n_G(H_12)
    fvec[13] = 1. / 12. * np.sum(np.sum(A2f * (A2 - np.ones(A2.shape)) * (A2 - 2. * np.ones(A2.shape))))  # n_G(H_13)
    named_features_vector = {associated_strings[i]:fvec[i] for i in range(14)} # generate a dictionary with associated counts
    print(named_features_vector)
    return fvec


def plotter(adjacency_matrix):
    G = nx.from_numpy_matrix(adjacency_matrix)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)

# ======================================================================================================================
# T2(H3)
# ======================================================================================================================
def H3_evolution():
    H3 = np.diag(np.ones(3), 1)

    H3_variant_1 = H3.copy()
    H3_variant_1[0,2] =1

    H3_variant_2 = H3.copy()
    H3_variant_2[0, 3] = 1

    #Finally ensure symmetry
    H3 += H3.T
    H3_variant_1 += H3_variant_1.T
    H3_variant_2 += H3_variant_2.T


    fig = plt.figure(figsize=(16,5))
    plt.subplot(131)
    plotter(H3)
    plt.subplot(132)
    plotter(H3_variant_1)
    plt.subplot(133)
    plotter(H3_variant_2)
    plt.savefig('TwitterMotifImages/H3_T2_evolution.png')

# ======================================================================================================================
# T2(H4)
# ======================================================================================================================
def H4_evolution():
    H4 = np.zeros((4, 4))
    H4[0, 1:4] = 1

    H4_variant_1 = H4.copy()
    H4_variant_1[1,2] = 1

    H4 += H4.T
    H4_variant_1 +=H4_variant_1.T

    subgraph_counter(H4)

    plt.figure(figsize=(11,5))
    plt.subplot(121)
    plotter(H4)
    plt.subplot(122)
    plotter(H4_variant_1)
    plt.savefig('TwitterMotifImages/H4_T2_evolution.png')

# ======================================================================================================================
# T2(H5)
# ======================================================================================================================
def H5_evolution():
    H5 = np.zeros((4, 4))
    H5[0, 1:4] = 1
    H5[1, 2] = 1

    H5_variant_1 = H5.copy()
    H5_variant_1[1,3] = 1
    H5 += H5.T
    H5_variant_1 += H5_variant_1.T

    plt.figure(figsize=(11, 5))
    plt.subplot(121)
    plotter(H5)
    plt.subplot(122)
    plotter(H5_variant_1)
    plt.savefig('TwitterMotifImages/H5_T2_evolution.png')

# ======================================================================================================================
# T2(H6)
# ======================================================================================================================
def H6_evolution():
    H6 = np.ones((4, 4)) - np.diag(np.ones(4))
    H6[1, 3] = 0
    H6[3, 1] = 0

    H6_variant_1 = np.ones((4, 4)) - np.diag(np.ones(4))
    plt.figure(figsize=(11, 5))
    plt.subplot(121)
    plotter(H6)
    plt.subplot(122)
    plotter(H6_variant_1)
    plt.savefig('TwitterMotifImages/H6_T2_evolution.png')

# ======================================================================================================================
# T2(H7)
# ======================================================================================================================
def H7_evolution():
    H7 = np.zeros((5, 5))
    H7[0, :5] = 1
    H7[1, 2] = 1

    H7_variant_1 = H7.copy()
    H7_variant_1[1,3] = 1

    H7_variant_2 = H7.copy()
    H7_variant_2[1,4] = 1

    H7 += H7.T
    H7_variant_1 += H7_variant_1.T
    H7_variant_2 += H7_variant_2.T

    plt.figure(figsize=(16, 5))
    plt.subplot(131)
    plotter(H7)
    plt.subplot(132)
    plotter(H7_variant_1)
    plt.subplot(133)
    plotter(H7_variant_2)
    plt.savefig('TwitterMotifImages/H7_T2_evolution.png')
# ======================================================================================================================
# T2(H8)
# ======================================================================================================================

def H8_evolution():
    H8 = np.zeros((5, 5))
    H8[0, 1:4] = 1
    H8[1, 2] = 1
    H8[1, 4] = 1

    H8_variant_1 = H8.copy()
    H8_variant_1[1,3] = 1
    H8_variant_2 = H8.copy()
    H8_variant_2[2, 4] = 1
    H8_variant_3 = H8.copy()
    H8_variant_3[3, 4] = 1

    H8 += H8.T
    H8_variant_1 += H8_variant_1.T
    H8_variant_2 += H8_variant_2.T
    H8_variant_3 += H8_variant_3.T

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plotter(H8)
    plt.subplot(222)
    plotter(H8_variant_1)
    plt.subplot(223)
    plotter(H8_variant_2)
    plt.subplot(224)
    plotter(H8_variant_3)
    plt.savefig('TwitterMotifImages/H8_T2_evolution.png')

H3_evolution()
H4_evolution()
H5_evolution()
H6_evolution()
H7_evolution()
H8_evolution()