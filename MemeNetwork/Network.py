import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# https://ulir.ul.ie/bitstream/handle/10344/4462/Gleeson_2014_competition.pdf;sequence=1
# https://lilianweng.github.io/lil-log/assets/papers/competition-limited-attention.pdf

####################################################################################################
##  Plotting Functions
####################################################################################################


def greatest_meme_subgraph_generator(adjacency_matrix, hot_encoder):

    meme = np.argmax(np.bincount(hot_encoder))
    G = nx.from_numpy_matrix(adjacency_matrix)
    elements = np.where(hot_encoder == meme)[0]
    meme_subgraph = G.subgraph(elements)
    return meme_subgraph, meme


def network_plotter(adjacency_matrix, hot_encoder):

    G = nx.from_numpy_matrix(adjacency_matrix)
    subgraph, meme = greatest_meme_subgraph_generator(adjacency_matrix, hot_encoder)
    nx.is_directed(G)
    pos = nx.circular_layout(G)

    plt.figure(figsize=(15,7))
    plt.subplot(121)
    nx.draw_networkx_edges(subgraph, pos=pos, alpha=0.2)
    nx.draw_networkx_nodes(subgraph, pos=pos, cmap='turbo', node_size=30)
    plt.title('Most Popular Meme Subgraph')

    plt.subplot(122)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.2, arrows=True, arrowsize=100)
    nx.draw_networkx_nodes(G, node_color= hot_encoder, pos=pos, cmap='turbo', node_size=30, alpha=0.8)
    plt.title('Network')
    plt.show()


def stats_info(adjacency_matrix, hot_encoder):

    histogram = np.histogram(hot_encoder)[0]

    plt.figure(figsize=(7,7))
    plt.subplot(121)
    plt.hist(hot_encoder, rwidth=0.8)
    plt.title('Frequency of Survived Memes')

    plt.subplot(122)
    plt.show()


def meme_evolution_plotter(history, meme_tracker):
    n = meme_tracker+1
    m = len(history)
    history_matrix = np.zeros((n,m)) + 0.05
    print(history_matrix.shape)
    for i in range(1,m):
        array = history[i]
        frequencies = np.bincount(array)
        history_matrix[0:frequencies.shape[0], i] = frequencies
    history_matrix = history_matrix[1:,1:]
    plt.pcolor(history_matrix.T, cmap=cm.viridis)
    plt.colorbar()
    plt.title('Popularity of Memes over Time')
    plt.show()


def plot_motifs(features_matrix):
    plt.figure(figsize=(10,10))
    labels = ['c3', 'c4', 'c5', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
    for i in range(9):
        plt.plot(features_matrix[i, :])
    plt.legend(labels)
    plt.show()


####################################################################################################
##  Subgraph Counting
####################################################################################################


def subgraph_counter(adjacency_matrix):
    G = nx.from_numpy_matrix(adjacency_matrix)
    features_vector = np.zeros(9)
    sz = G.number_of_nodes()
    degrees = np.zeros(sz)
    ovec = np.zeros(sz)
    dfvec = np.zeros(sz)
    dftvec = np.zeros(sz)

    degrees[:] = np.array([d for n, d in G.degree()])
    ovec[:] = np.ones(degrees.size)
    dfvec[:] = degrees[:] - ovec[:]
    dftvec[:] = degrees[:] - 2. * ovec[:]

    A = nx.adjacency_matrix(G)
    Af = A.todense()
    A2 = Af @ Af
    # Q = np.multiply(A2,A)
    A3 = Af @ A2
    A4 = Af @ A3
    A5 = Af @ A4
    A6 = Af @ A5
    A7 = Af @ A6
    dfA = np.multiply(np.diag(dfvec) @ Af, Af @ np.diag(dfvec))
    dftA = np.multiply(np.diag(dftvec) @ Af, Af @ np.diag(dftvec))
    P1 = G.number_of_edges()
    P2 = np.sum(np.multiply(degrees, dfvec)) / 2.

    # note, first three entries of fvec are the number of cycles C_3, C_4, and C_5

    features_vector[0] = np.trace(A3) / 6.  # n_G(C_3)
    features_vector[1] = (np.trace(A4) - 4. * P2 - 2. * P1) / 8.  # n_G(C_4)
    features_vector[3] = np.sum(np.sum(dfA)) - 3. * features_vector[0]  # n_G(H_3)
    features_vector[4] = np.dot(np.multiply(degrees, dfvec), dftvec) / 6.  # n_G(H_4)
    features_vector[5] = np.dot(np.diag(A3), dftvec) / 2.  # n_G(H_5)
    features_vector[6] = 1 / 2 * np.sum(np.sum(np.multiply(np.multiply((A2 - np.ones((A2.shape))), A2), Af)))  # n_G(H_6)
    features_vector[7] = 1 / 4 * np.dot(np.diag(A3), np.multiply(dfvec, dftvec - np.ones(dftvec.size)))  # n_G(H_7)
    features_vector[8] = np.sum(np.sum(np.multiply(np.multiply(A2, dftA), Af))) - 2 * features_vector[6]  # n_G(H_8)
    features_vector[2] = (np.trace(A5) - 10 * features_vector[5] - 30 * features_vector[0]) / 10.  # n_G(C_5)

    return features_vector


####################################################################################################
##  Meme Simulator Functions
####################################################################################################


def node_spawn(adjacency_matrix, hot_encoder_array, l, gamma):

    node_arrival = np.random.choice(['new_node', 'no_node'], p=[l, 1 - l])
    if node_arrival == 'new_node':  # could make this pref-attatched

        # generate new adjacenecy matrix
        n, m = adjacency_matrix.shape
        temporary_matrix = np.zeros((n + 1, n + 1))  # This is wildly ineffecient, need to fix
        temporary_matrix[:n, :n] = adjacency_matrix
        adjacency_matrix = temporary_matrix

        ## Generate new hot_encoder_array
        hot_encoder_array = np.append(hot_encoder_array, np.random.choice(
            hot_encoder_array))  # choose a possible a meme from existing memes for the new user to view

        # A new arrival will most likely follow those aready, but is quite not ready yet to be followed
        adjacency_matrix[:, -1] = np.random.rand(n + 1)  # choose refollows
        adjacency_matrix[adjacency_matrix < 1 - gamma] = 0
        adjacency_matrix = np.round(adjacency_matrix)

        adjacency_matrix[-1, :] = np.random.rand(n + 1)  # gains followers
        adjacency_matrix[adjacency_matrix < 1 - gamma] = 0
        adjacency_matrix = np.round(adjacency_matrix)
        adjacency_matrix[-1, -1] = 0
    return adjacency_matrix, hot_encoder_array


def preferential_node_spawn(adjacency_matrix, hot_encoder_array,l):
    node_arrival = np.random.choice(['new_node', 'no_node'], p=[l, 1 - l])
    if node_arrival == 'new_node':
        source_node_probability = np.sum(adjacency_matrix, axis=0)/np.sum(adjacency_matrix)
        print(np.sum(source_node_probability))
        #print(source_node_probability)

        # This line below only allows for one attachent per new node which is not expressly the preferential attachment model
        # source_node = np.random.choice(np.arange(adjacency_matrix.shape[0]), p=source_node_probability)
        n, m = adjacency_matrix.shape
        temporary_matrix = np.zeros((n + 1, n + 1))
        temporary_matrix[:n, :n] = adjacency_matrix
        adjacency_matrix = temporary_matrix
        for i in range(n):
            if np.random.rand(1) <= source_node_probability[i]:
                adjacency_matrix[i, -1] = 1
        hot_encoder_array = np.append(hot_encoder_array, np.random.choice(hot_encoder_array))
    return adjacency_matrix, hot_encoder_array


def preferential_attachment_event(adjacency_matrix):
    """
    :param adjacency_matrix:
    :return: adjacency matrix with new attachments formed based on preferential attachment mechanism. Preserves
                number of nodes.
    """
    source_node_probability = np.sum(adjacency_matrix, axis=0)/np.sum(adjacency_matrix)
    source_node = np.random.choice(np.arange(adjacency_matrix.shape[0]), p=source_node_probability)
    target_nodes = np.delete(np.arange(adjacency_matrix.shape[0]), source_node)
    target_node = np.random.choice(target_nodes)
    adjacency_matrix[source_node,target_node] = 1
    return adjacency_matrix


def meme_spawn(adjacency_matrix, meme_tracker, hot_encoder_array, mu):
    meme_arrival = np.random.choice(['new_meme', 'current_meme'], p=[mu, 1 - mu])
    if meme_arrival == 'new_meme':
        meme_tracker += 1
        array_length = hot_encoder_array.shape[0]
        user = np.random.randint(0, array_length, 1)
        hot_encoder_array[user] = meme_tracker
        followers = np.where(adjacency_matrix[user,:] == 1)[1]

    else:
        array_length = hot_encoder_array.shape[0]
        user = np.random.randint(0, array_length, 1)
        followers = np.where(adjacency_matrix[int(user), :] == 1)
        for follower in followers:
            hot_encoder_array[follower] = meme_tracker

    return adjacency_matrix, hot_encoder_array, meme_tracker


def matrix_initializer(n, populations, initial_memes, gamma):
    adjacency_matrix = np.zeros((n * populations, n * populations))
    for pop in range(populations):
        adjacency_submatrix = np.random.rand(n,n)
        adjacency_submatrix[adjacency_submatrix < 1 - gamma] = 0
        adjacency_submatrix = np.round(adjacency_submatrix)
        adjacency_matrix[n*pop:n*(pop+1),n*pop:n*(pop+1)] = adjacency_submatrix

    # we want always at least some connect
    adjacency_matrix[1,-1] = 1
    adjacency_matrix[-1,1] = 1

    adjacency_matrix -= np.diag(np.diag(adjacency_matrix)) # Eliminate self-loops
    hot_encoder_array = np.random.randint(1, initial_memes+1, n*populations) # Is the node red or blue?
    meme_tracker = np.max(hot_encoder_array)
    return adjacency_matrix, hot_encoder_array, meme_tracker


def network_run(n=20, populations=2, initial_memes=3, mu=0.04, l=0.0, gamma=0.05, maximum_time=100,
                preferential_attachment=True):

    """
    :param n: Users per population
    :param populations: Defines number of communities in Network. They are initially unconncected
    :param initial_memes: self-explanatory
    :param mu: probability between 0 and 1  a new meme enters at time t
    :param l: probability between 0 and 1 a node is spawned at time t
    :param gamma: probability between 0 and 1 at event a node is connected
    :param maximum_time: self-explanatory
    :param preferential_attachment: specify a preferential attatchment mechanism as nodes are added
    :return: network adjacency matrix, and meme hot encoder
    """

    adjacency_matrix, hot_encoder_array, meme_tracker = matrix_initializer(n, populations, initial_memes, gamma)
    history = []
    features_matrix = np.zeros((9,maximum_time))

    if preferential_attachment:
        for t in range(maximum_time):
            features_matrix[:,t] = subgraph_counter(adjacency_matrix)
            history.append(hot_encoder_array.copy())
            adjacency_matrix, hot_encoder_array = preferential_node_spawn(adjacency_matrix, hot_encoder_array, l)
            adjacency_matrix, hot_encoder_array, meme_tracker = meme_spawn(adjacency_matrix, meme_tracker,
                                                                           hot_encoder_array, mu)
            #adjacency_matrix = preferential_attachment_event(adjacency_matrix)
    else:
        for t in range(maximum_time):
            adjacency_matrix, hot_encoder_array = node_spawn(adjacency_matrix, hot_encoder_array, l, gamma)
            adjacency_matrix, hot_encoder_array, meme_tracker = meme_spawn(adjacency_matrix, meme_tracker,
                                                                           hot_encoder_array, mu)

    return adjacency_matrix, hot_encoder_array, meme_tracker, history, features_matrix


####################################################################################################
## Run the Thing
####################################################################################################
# H3, H8's heavily correlate for meme dominance - cluster with fan pattern around
# Higher initial populations generate more h7's which then closley track H3, but there is a high abundance of

if __name__ == '__main__':
    adj_mat, hot_array, meme_tracker, history, features_matrix = network_run()
    network_plotter(adj_mat, hot_array)
    #stats_info(adj_mat, hot_array)
    meme_evolution_plotter(history, meme_tracker)
    plt.scatter(range(9),features_matrix[:,0])
    plt.show()
    print('We cycled through a total of {} memes.' .format(meme_tracker))
