import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from DynamicModeDecomposition.DMD import standard_dynamic_mode_decomposition
from DynamicModeDecomposition.plotter import eigen_plot, data_plot
from DynamicModeDecomposition.subgraph_counter import subgraph_count
import pickle

def matrix_to_features():
    file_name = '../Data/preferential_attachment_model_with_params_10_2_100.p'
    data = np.array(pickle.load(open(file_name, 'rb')))

    features_matrix = np.zeros((14, data.shape[0]-10))
    for t in range(10, data.shape[0]):
        features_matrix[:, t-10] = subgraph_count(data[:t, :t])

    features_file_name = '../Data/features_pref_attach_params_10_2_100.p'
    pickle.dump(features_matrix, open(features_file_name, 'wb'))

# matrix_to_features()

file_name = '../Data/features_pref_attach_params_10_2_100.p'
features_matrix = np.array(pickle.load(open(file_name, 'rb')))
data_plot(features_matrix)

eigenvectors, eigenvalues, phi_modes, data_average = standard_dynamic_mode_decomposition(features_matrix, 4)
eigen_plot(eigenvectors, eigenvalues, phi_modes.T)