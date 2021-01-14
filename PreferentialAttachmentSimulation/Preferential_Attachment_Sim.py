import numpy as np
import pickle

def preferential_attachment(start_nodes,m,end_nodes, save_as_pickle=True):
    """
    :param start_nodes: integer obvi
    :param m:  number of edges to add each iteration
    :param end_nodes: integer > start_nodes obvi
    :param save_as_pickle: bool obvi. True saves as pickle file
    :return: Adjacency matrix of a preferential attachment graph
    """

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

    if save_as_pickle:
        file_name = '../Data/preferential_attachment_model_with_params_'+ str(start_nodes) + \
            '_' + str(m) + '_' + str(end_nodes) + '.p'
        pickle.dump(adjacency_matrix, open(file_name, 'wb'))

    return adjacency_matrix

adjacency_matrix = preferential_attachment(10, 2, 100, save_as_pickle=True)