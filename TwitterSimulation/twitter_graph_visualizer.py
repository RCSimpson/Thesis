import numpy as np
import networkx as nx
import collections
import matplotlib.pyplot as plt

#these functions have to run concurent to the actual simulation, because of the T3 events

def histogram(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse = True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    return deg, cnt

def early_graph_snapshots(adjacency_matrix, time, time_step=2, max_time=8):
    time_steps = range(1, max_time+1, time_step)

    for timing in time_steps:
        if time != timing:
            break
        else:
            G = nx.from_numpy_matrix(adjacency_matrix)

    return None

def graph_stats(adjacency_matrix,t, max_time):
    if t == 0:
        node_count = np.zeros(max_time)
        edge_count = np.zeros(max_time)
        edge_density = np.zeros(max_time)
        max_node_degree = np.zeros(max_time)
    G = nx.from_numpy_matrix(adjacency_matrix)
    H = G.undirected()

    node_count[t] = nx.number_of_nodes(H)
    edge_count[t] = nx.number_of_edges(H)
    edge_count[t] = nx.density(H)
    max_node_degree[t] = max([val for (node, val) in H.degree()])

    if t==max_time:
        plt.subplot(221)
        plt.plot(node_count, color='b', label='Number of Edges')
        plt.plot(edge_count, color='orange', label='Number of Nodes')
        plt.title('Graph Development')
        plt.legend()

        plt.subplot(222)
        plt.plot(edge_density, color='orange', label='Edge Density')
        plt.title('Edge Density')

        plt.subplot(223)
        plt.plot(max_node_degree, color='orange', label='Greatest Node Degree')
        plt.title('Greatest Degree')

        plt.subplot(224)
        deg, cnt = histogram(G)
        plt.scatter(deg, cnt)
        plt.title('Final Time Degree Histogram')
        plt.savefig('twitter_sim.png') #Make sure its unique
    return None