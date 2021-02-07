import numpy as np
import networkx as nx
from DynamicModeDecomposition.subgraph_counter import subgraph_count
import matplotlib.pyplot as plt

#for each root_node we have an associated component.
#before selecting the root or what have you we need to select the correct tree.

def networkx_tree_chooser(adjacency_matrix, root_node_list):
    nodes = 0
    probabilities = 0
    G = nx.from_numpy_matrix(adjacency_matrix)

    return nodes, probabilities

def simple_tree_chooser(adjacency_matrix, root_node_list):
    probs = []
    node_choice = []
    for node in root_node_list:
        if np.int(np.sum(adjacency_matrix[node, :]) + np.sum(adjacency_matrix[:, node])) > 0:
            probs.append(np.int(np.sum(adjacency_matrix[node, :]) + 1))
            node_choice.append(node)
    return node_choice, probs/(np.sum(probs))

def run_simulation(n,l,p):
    '''
    :param n: number of initial nodes
    :param l: for high l increase likelihood of root node spawn
    :param p: for high p increased likelihood of node spawn + attachment, for low p high increase of attachment w/o node spawn
    :return: matrix with motif counts over time
    '''

    p1 = 0.3 #l/(l+1)
    p2 = 0.4 # p/(l+1)
    p3 = 0.3 # (1-p)/(l+1)
    q = 0.9

    if p1+p2+p3 != 1:
        print('Probabilities must equal one')
        return None

    root_node_array= np.array([0])
    tmax = 300
    features_matrix = np.zeros((14, tmax))
    adjacency_matrix = np.random.rand(n, n)
    adjacency_matrix[0,:]=1
    adjacency_matrix[adjacency_matrix < 1 - 0.8] = 0
    adjacency_matrix[adjacency_matrix >= 1 - 0.8] = 1

    t=0
    while t < tmax:
        event = np.random.choice(['t1', 't2', 't3'], p=[p1, p2, p3])
        N, N = adjacency_matrix.shape

        if event == 't1':
            #Generate new node without attachments
            temporary_matrix = np.zeros((N+1, N+1))
            temporary_matrix[:N, :N] = adjacency_matrix
            adjacency_matrix = temporary_matrix
            root_node_array = np.append(root_node_array, N)

        if event == 't2':
            #Generate new node and new attachmen
            #target = new node retweets source
            temporary_matrix = np.zeros((N+1, N+1))
            temporary_matrix[:N, :N] = adjacency_matrix
            adjacency_matrix = temporary_matrix
            source_type_node = np.random.choice(['root', 'other'], p=[q, 1-q])

            if source_type_node=='root':
                nodes, probabilities = simple_tree_chooser(adjacency_matrix, root_node_array)
                source_node = np.random.choice(nodes, p=probabilities)
            else:
                source_node = np.random.choice(N)
            adjacency_matrix[source_node, N] = 1

        if event=='t3':
            #Generate new node between edges
            source_type_node = np.random.choice(['root', 'other'], p=[q, 1 - q])
            target_node = np.random.choice(N-1)

            if source_type_node=='root':
                nodes, probabilities = simple_tree_chooser(adjacency_matrix,root_node_array)
                source_node = np.random.choice(nodes, p=probabilities)
            else:
                source_node = np.random.choice(N-1)
            adjacency_matrix[source_node, target_node] = 1

        # undirected_adjacency_matrix -= np.diag(np.diag(undirected_adjacency_matrix))
        adjacency_matrix -= np.diag(np.diag(adjacency_matrix))
        undirected_adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        undirected_adjacency_matrix[undirected_adjacency_matrix > 0] = 1
        features_matrix[:, t] = subgraph_count(undirected_adjacency_matrix)

        t+=1

    return features_matrix

ftmat = run_simulation(n=3, l=0.1, p=0.1)

associated_strings = ['C3', 'C4', 'C5', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13']
colors = ['r','b','g','c','y','k', 'orange', 'm', 'salmon','sienna', 'gray','plum','orangered','navy']
linestyles = ['--',':','-','--',':','-','--',':','-','--',':','-','--',':']

for i in range(14):
    plt.plot(ftmat[i,:], color=colors[i])
plt.legend(associated_strings)
plt.show()

