import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# ======================================================================================================================
# Preferential Attachment Model
# ======================================================================================================================
# ======================================================================================================================

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
# H3
# ======================================================================================================================

def H3_evolution():
    H3 = np.diag(np.ones(3), 1)
    H3 += H3.T
    #root attachment
    H3_root = np.diag(np.ones(4), 1)
    H3_root += H3_root.T
    #non-root attachement
    H3_non_root = np.zeros((5,5))
    H3_non_root[:4,:4] = np.diag(np.ones(3), 1)
    H3_non_root[4,2] = 1
    H3_non_root += H3_non_root.T
    #root and non-root attachment
    #H3_double = np.diag(np.ones(3), 1)
    #H3_double[0,1] = 1

    subgraph_counter(H3,)
    subgraph_counter(H3_root)
    subgraph_counter(H3_non_root)

    plt.figure(figsize=(16,5))
    plt.subplot(131)
    plotter(H3)

    plt.subplot(132)
    plotter(H3_root)

    plt.subplot(133)
    plotter(H3_non_root)
    plt.savefig('PrefMotifImages/H3_evolution.png')

# ======================================================================================================================
# H4
# ======================================================================================================================

def H4_evolution():
    H4 = np.zeros((4, 4))
    H4[0, 1:4] = 1
    H4 += H4.T

    #root attachment
    H4_root = np.zeros((5,5))
    H4_root[0, 1:5] = 1
    H4_root += H4_root.T

    #non-root attachement
    H4_non_root = np.zeros((5, 5))
    H4_non_root[0, 1:4] = 1
    H4_non_root[3,4]=1
    H4_non_root += H4_non_root.T

    #root and non-root attachment
    H4_double = np.diag(np.ones(3), 1)
    H4_double[0,1] = 1

    subgraph_counter(H4)

    subgraph_counter(H4_root)

    subgraph_counter(H4_non_root)

    plt.figure(figsize=(16,5))
    plt.subplot(131)
    plotter(H4)
    plt.subplot(132)
    plotter(H4_root)
    plt.subplot(133)
    plotter(H4_non_root)
    plt.savefig('PrefMotifImages/H4_evolution.png')

# ======================================================================================================================
# H5
# ======================================================================================================================

def H5_evolution():

    H5 = np.zeros((4, 4))
    H5[0, 1:4] = 1
    H5[1, 2] = 1
    H5 += H5.T

    # attachment 1
    H5_root = np.zeros((5, 5))
    H5_root[0, 1:4] = 1
    H5_root[1, 2] = 1
    H5_root[2, 4] = 1
    H5_root += H5_root.T

    # attachment 2
    H5_1 = np.zeros((5, 5))
    H5_1[0, 1:4] = 1
    H5_1[1, 2] = 1
    H5_1[3, 4] = 1
    H5_1 += H5_1.T
    print(subgraph_counter(H5))

    print(subgraph_counter(H5_root))

    print(subgraph_counter(H5_1))

    fig = plt.figure(figsize=(16, 5))
    plt.subplot(131)
    plotter(H5)
    plt.subplot(132)
    plotter(H5_root)
    plt.subplot(133)
    plotter(H5_1)
    plt.savefig('H5_evolution.png')

# ======================================================================================================================
# H6
# ======================================================================================================================

def H6_evolution():
    H6 = np.ones((4, 4)) - np.diag(np.ones(4))
    H6[1, 3] = 0
    H6[3, 1] = H6[1, 3]

    # attachment 1
    H6_1 = np.zeros((5,5))
    H6_1[:4,:4]  = H6
    H6_1[3, 1] = H6_1[1, 3]
    H6_1[1,4] = 1
    H6_1[4,1] =  H6_1[1,4]

    # attachment 2
    H6_2 = np.zeros((5, 5))
    H6_2[:4, :4] = H6
    H6_2[3, 1] = H6_2[1, 3]
    H6_2[2, 4] = 1
    H6_2[4, 2] = H6_2[1, 4]

    fig = plt.figure(figsize=(16, 5))
    plt.subplot(131)
    plotter(H6)
    plt.subplot(132)
    plotter(H6_1)
    plt.subplot(133)
    plotter(H6_2)
    plt.savefig('PrefMotifImages/H6_evolution.png')

# ======================================================================================================================
# H7
# ======================================================================================================================

def H7_evolution():
    H7 = np.zeros((5, 5))
    H7[0, :5] = 1
    H7[1, 2] = 1
    H7[2, 1] = 1

    # attachment 1
    H7_1 = np.zeros((6,6))
    H7_1[:5,:5]  = H7
    H7_1[3, 1] = H7_1[1, 3]
    H7_1[1,5] = 1
    H7_1[5,1] =  1

    # attachment 2
    H7_2 = np.zeros((6, 6))
    H7_2[:5, :5] = H7
    H7_2[3, 1] = H7_2[1, 3]
    H7_2[3, 5] = 1
    H7_2[5, 3] = H7_2[3, 5]

    # attachment 3
    H7_3 = np.zeros((6, 6))
    H7_3[:5, :5] = H7
    H7_3[3, 1] = H7_3[1, 3]
    H7_3[0, 5] = 1
    H7_3[5, 0] = 1

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plotter(H7)

    plt.subplot(222)
    plotter(H7_1)

    plt.subplot(223)
    plotter(H7_2)

    plt.subplot(224)
    plotter(H7_3)
    plt.savefig('H7_evolution.png')

# ======================================================================================================================
# H8
# ======================================================================================================================

def H8_evolution():

    H8 = np.zeros((5, 5))
    H8[0, :4] = 1
    H8[1, 2] = 1
    H8[2, 1] = 1
    H8[1, 4] = 1
    H8[4, 1] = 1

    # attachment 1
    H8_1 = np.zeros((6,6))
    H8_1[:5,:5] = H8
    H8_1[1,5] = 1
    H8_1[5,1] = H8_1[1,4]

    # attachment 2
    H8_2 = np.zeros((6, 6))
    H8_2[:5, :5] = H8
    H8_2[2, 5] = 1
    H8_2[5, 2] = H8_2[1, 4]

    # attachment 3
    H8_3 = np.zeros((6, 6))
    H8_3[:5, :5] = H8
    H8_3[3, 5] = 1
    H8_3[5, 3] = H8_3[1, 4]

    fig = plt.figure(figsize=(10,10))
    plt.subplot(221)
    plotter(H8)
    plt.subplot(222)
    plotter(H8_1)
    plt.subplot(223)
    plotter(H8_2)
    plt.subplot(224)
    plotter(H8_3)
    plt.savefig('PrefMotifImages/H8_evolution.png')


#H3_evolution()
# H4_evolution()
# H5_evolution()
# H6_evolution()
# H7_evolution()
# H8_evolution()



