import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from DynamicModeDecomposition.DMD import standard_dynamic_mode_decomposition
from DynamicModeDecomposition.plotter import eigen_plot, data_plot
from DynamicModeDecomposition.subgraph_counter import subgraph_count
import pickle

file_name = '../Data/features_matrix_sample.p'
features_matrix = np.array(pickle.load(open(file_name, 'rb')))
data_plot(features_matrix[:,:-2])

eigenvectors, eigenvalues, phi_modes, data_average = standard_dynamic_mode_decomposition(features_matrix[:,:-2], 4)
eigen_plot(eigenvectors, eigenvalues, phi_modes.T)