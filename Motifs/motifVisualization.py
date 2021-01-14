import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# A collection of Python functions to generate examples of definitions pertinent to the Thesis.

# ======================================================================================================================
# Homomorphism Example
# ======================================================================================================================

def homomoprhism():
    labels = {}
    labels[0] = 'A'
    labels[1] = 'B'
    labels[2] = 'C'
    labels[3] = 'D'
    A = np.ones((4,4)) - np.diag(np.diag(np.ones((4,4))))
    A[0,1] = 0
    A[1,0] = 0

    B = np.ones((3,3)) - np.diag(np.diag(np.ones((3,3))))

    fig = plt.figure(figsize=(10,5))
    plt.subplot(121)
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_labels(G, pos=pos)
    plt.title('$G$')

    plt.subplot(122)
    H = nx.from_numpy_matrix(B)
    pos = nx.spring_layout(H)
    nx.draw_networkx_edges(H, pos=pos)
    nx.draw_networkx_nodes(H,pos=pos, node_color='c')
    nx.draw_networkx_labels(H, pos=pos, labels={0:'A', 1:'B', 2:'C'})
    plt.title('$H$')
    plt.savefig('graph_homomorphism.png')

# ======================================================================================================================
# Isomorphism Example
# ======================================================================================================================

def isomorphism():
    labels = {}
    labels[0] = 'A'
    labels[1] = 'B'
    labels[2] = 'C'
    A = np.ones((4,4)) - np.diag(np.diag(np.ones((4,4))))
    A[0,1] = 0
    A[1,0] = 0

    B = A

    fig = plt.figure(figsize=(10,5))
    plt.subplot(121)
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_labels(G, pos=pos)
    plt.title('$G$')

    plt.subplot(122)
    H = nx.from_numpy_matrix(B)
    pos = nx.circular_layout(H)
    nx.draw_networkx_edges(H, pos=pos)
    nx.draw_networkx_nodes(H,pos=pos, node_color='c')
    nx.draw_networkx_labels(H, pos=pos, labels=labels)
    plt.title('$H$')
    plt.savefig('graph_isomoprhism.png')

# ======================================================================================================================
# Automorphism Example
# ======================================================================================================================

def automorphism():

    pass

# ======================================================================================================================
# Subgraph and Induced Subgraph Example
# ======================================================================================================================

def subgraph():

    A = np.ones((5,5)) - np.diag(np.diag(np.ones((5,5))))
    A[2,3] = 0
    A[3,2] = 0
    A[3,4] = 0
    A[4,3] = 0
    A[3,0] = 0
    A[0,3] = 0
    A[2,4] = 0
    A[4,2] = 0
    A[0,4] = 0
    A[4,0] = 0

    subgraph_A = np.ones((3,3)) - np.diag(np.diag(np.ones((3,3))))
    subgraph_A[1,2] = 0
    subgraph_A[2,1] = 0

    induced_subgraph_A = np.ones((3,3)) - np.diag(np.diag(np.ones((3,3))))

    fig = plt.figure(figsize=(15,5))
    plt.subplot(131)
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_labels(G, pos=pos)
    plt.title('$G$')

    plt.subplot(132)
    H = nx.from_numpy_matrix(subgraph_A)
    #pos = nx.circular_layout(H)
    nx.draw_networkx_edges(H, pos=pos)
    nx.draw_networkx_nodes(H,pos=pos, node_color='c')
    nx.draw_networkx_labels(H, pos=pos)
    plt.title('Subgraph of $G$')

    plt.subplot(133)
    HH = nx.from_numpy_matrix(induced_subgraph_A)
    #pos = nx.circular_layout(HH)
    nx.draw_networkx_edges(HH, pos=pos)
    nx.draw_networkx_nodes(HH,pos=pos, node_color='c')
    nx.draw_networkx_labels(HH, pos=pos)
    plt.title('Induced Subgraph of $G$')

    plt.savefig('subgraph.png')

# ======================================================================================================================
# Motifs
# ======================================================================================================================

def motif_set_one():
    fig = plt.figure(figsize = (10,10))
    plt.subplot(221)
    H3 = np.diag(np.ones(3), 1)
    G = nx.from_numpy_matrix(H3)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos,node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H3', size=20)

    plt.subplot(222)
    H4 = np.zeros((4,4))
    H4[0,1:4]=1
    G = nx.from_numpy_matrix(H4)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H4', size=20)

    plt.subplot(223)
    H5 = np.zeros((4,4))
    H5[0,1:4]=1
    H5[1,2] = 1
    H5[2,1] = 1
    G = nx.from_numpy_matrix(H5)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos,node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H5', size=20)

    plt.subplot(224)
    H6 = np.ones((4,4)) - np.diag(np.ones(4))
    H6[1,3]=0
    H6[3,1]=0
    G = nx.from_numpy_matrix(H6)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H6', size=20)
    plt.savefig('GeneralMotifImages/motif_set_one.png')

def motif_set_two():
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(221)
    H7 = np.zeros((5,5))
    H7[0,:5] = 1
    H7[1,2] = 1
    H7[2,1] = 1
    G = nx.from_numpy_matrix(H7)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H7', size=20)

    plt.subplot(222)
    H8 = np.zeros((5, 5))
    H8[0, :4] = 1
    H8[1, 2] = 1
    H8[2, 1] = 1
    H8[1, 4] = 1
    H8[4, 1] = 1
    G = nx.from_numpy_matrix(H8)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H8', size=20)

    plt.subplot(223)
    H9 = np.diag(np.ones(4),1)
    H9[4,1] = 1
    H9 += H9.T
    G = nx.from_numpy_matrix(H9)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H9', size=20)

    plt.subplot(224)
    H10 = np.diag(np.ones(3),2)
    H10[0,4] = 1
    H10[0,3] = 1
    H10 += H10.T
    print(H10)
    G = nx.from_numpy_matrix(H10)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color='c')
    nx.draw_networkx_edges(G, pos=pos)
    plt.title('H10', size=20)

    plt.savefig('GeneralMotifImages/motif_set_two.png')


motif_set_one()
motif_set_two()