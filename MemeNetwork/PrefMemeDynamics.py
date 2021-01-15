import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def network_plotter(adjacency_matrix, hot_encoder):

    G = nx.from_numpy_matrix(adjacency_matrix)
    # subgraph, meme = greatest_meme_subgraph_generator(adjacency_matrix, hot_encoder)
    nx.is_directed(G)
    pos = nx.spring_layout(G)

    plt.figure(figsize=(15,7))
    # plt.subplot(121)
    # nx.draw_networkx_edges(subgraph, pos=pos, alpha=0.2)
    # nx.draw_networkx_nodes(subgraph, pos=pos, cmap='turbo', node_size=30)
    # plt.title('Most Popular Meme Subgraph')

    plt.subplot(122)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.2, arrows=True, arrowsize=100)
    nx.draw_networkx_nodes(G, node_color= hot_encoder, pos=pos, cmap='turbo', node_size=30, alpha=0.8)
    plt.title('Network')
    plt.show()


def preferential_attachment(start_nodes,m,end_nodes):
    adjacency_matrix = np.zeros((end_nodes,end_nodes), dtype=np.int)
    percent = 0.75
    for i in range(1,start_nodes):
        for j in range(i):
            probability = np.random.uniform(0.,1.,1)
            if probability < percent:
                adjacency_matrix[i,j]=1
                adjacency_matrix[j,i]=1

    adjacency_matrix -= np.diag(np.diag(adjacency_matrix))
    probabilities = np.zeros(end_nodes,dtype=np.float64)

    for j in range(start_nodes,end_nodes):
        degrees = np.sum(adjacency_matrix,axis=1)
        total_degree = np.sum(adjacency_matrix)
        probabilities[:j] = degrees[:j]/total_degree
        distribution = np.ones(j, dtype=np.float64) #proper name?

        for l in range(1,j):
            distribution[l-1] =  np.sum(probabilities[:l])
        node_index = np.arange(j)

        for l in range(m):
            probability = np.random.uniform(0.,1.0,1)
            index = distribution > probability
            indexx = np.min(node_index[index])

            while adjacency_matrix[j,indexx] ==1:
                distribution[indexx] = 0
                probability =  np.random.uniform(0.,1.,1)
                index = distribution > probability
                indexx = np.min(node_index[index])

            adjacency_matrix[j, indexx] = 1
            adjacency_matrix[indexx, j] = 1
        adjacency_matrix[j,j]=0

    return adjacency_matrix

def meme_evolution_plotter(history, meme_tracker):
    n = meme_tracker+1
    m = len(history)
    history_matrix = np.zeros((n,m))

    for i in range(1,m):
        array = history[i]
        frequencies = np.bincount(array)
        history_matrix[0:frequencies.shape[0], i] = frequencies
    history_matrix = history_matrix[1:,1:]
    plt.pcolor(history_matrix.T, cmap=cm.viridis)
    plt.colorbar()
    plt.title('Popularity of Memes over Time')
    plt.show()

def meme_spawn(adjacency_matrix, meme_tracker, hot_encoder_array, mu):
    meme_arrival = np.random.choice(['new_meme', 'current_meme'], p=[mu, 1 - mu])
    if meme_arrival == 'new_meme':
        meme_tracker += 1
        array_length = hot_encoder_array.shape[0]
        user = np.random.randint(0, array_length, 1)
        hot_encoder_array[user] = meme_tracker

    else:
        array_length = hot_encoder_array.shape[0]
        user = np.random.randint(0, array_length, 1)
        followers = np.where(adjacency_matrix[int(user), :] == 1)

        for follower in followers:
            hot_encoder_array[follower] = meme_tracker

    return adjacency_matrix, hot_encoder_array, meme_tracker

def meme_propagation_pref_attachment(graph_size, initial_memes, meme_probability, max_time):
    adjacency_matrix = preferential_attachment(5, 2, graph_size) # intialize matrix
    hot_encoder = np.random.randint(1, initial_memes+1, graph_size)
    meme_tracker = initial_memes + 1
    history = []

    for t in range(max_time):
        history.append(hot_encoder.copy())
        adjacency_matrix, hot_encoder, meme_tracker = meme_spawn(adjacency_matrix, meme_tracker, hot_encoder, meme_probability)

    return adjacency_matrix, hot_encoder, history, meme_tracker

adjacency_matrix, hot_encoder, history, meme_tracker = meme_propagation_pref_attachment(100, 2, 0.2, 100)

#meme_evolution_plotter(history, meme_tracker)
network_plotter(adjacency_matrix, hot_encoder)
