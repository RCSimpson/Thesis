import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

#file_name = '../Data/pref_attach/preferential_attachment_model_with_params_10_2_300-20210118-163351.p'
file_name = '../Data/twitter_sim/090909_twitter_graph.p'
A = pickle.load(open(file_name, 'rb'))
#G = nx.from_numpy_matrix(A)
G=A
#centrality measures
eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000)
betweeness_centrality = nx.betweenness_centrality(G)
degree_centrality = nx.degree_centrality(G)
subgraph_centrality = nx.subgraph_centrality(G)
harmonic_centrality = nx.harmonic_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

fig = plt.figure(figsize=(10,10))
plt.subplot(321)
pos = nx.spring_layout(G, k=0.15, iterations=20)
eigen_centrality_list = [eigen_centrality[i] for i in range(len(eigen_centrality))]
scalar = (300/max(eigen_centrality_list))
eigen_size_list = [(scalar)*eigen_centrality[i] for i in range(len(eigen_centrality))]
nx.draw_networkx_nodes(G, pos,node_color=eigen_centrality_list, node_size=eigen_size_list)
nx.draw_networkx_edges(G, pos, width=0.1)
plt.title('Eigenvector Centrality')

plt.subplot(322)
betweeness_centrality_list = [betweeness_centrality[i] for i in range(len(betweeness_centrality))]
scalar = (300/max(betweeness_centrality_list))
betweness_size_list = [scalar*betweeness_centrality[i] for i in range(len(betweeness_centrality))]
nx.draw_networkx_nodes(G, pos,node_color=betweeness_centrality_list, node_size=betweness_size_list)
nx.draw_networkx_edges(G, pos, width=0.1)
plt.title('Betweeness Centrality')

plt.subplot(323)
degree_centrality_list = [degree_centrality[i] for i in range(len(degree_centrality))]
scalar = (300/max(degree_centrality_list))
degree_size_list = [scalar*degree_centrality[i] for i in range(len(degree_centrality))]
nx.draw_networkx_nodes(G, pos,node_color=degree_centrality_list, node_size=degree_size_list)
nx.draw_networkx_edges(G, pos, width=0.1)
plt.title('Degree Centrality')

plt.subplot(324)
subgraph_centrality_list = [subgraph_centrality[i] for i in range(len(subgraph_centrality))]
scalar = (300/max(subgraph_centrality_list))
subgraph_size_list = [scalar*subgraph_centrality[i] for i in range(len(subgraph_centrality))]
nx.draw_networkx_nodes(G, pos,node_color=subgraph_centrality_list, node_size=subgraph_size_list)
nx.draw_networkx_edges(G, pos, width=0.1)
plt.title('Subgraph Centrality')

plt.subplot(325)
harmonic_centrality_list = [harmonic_centrality[i] for i in range(len(harmonic_centrality))]
scalar = 100/max(harmonic_centrality_list)
harmonic_size_list = [scalar*harmonic_centrality[i] for i in range(len(harmonic_centrality))]
nx.draw_networkx_nodes(G, pos,node_color=harmonic_centrality_list, node_size=harmonic_size_list)
nx.draw_networkx_edges(G, pos, width=0.1)
plt.title('Harmonic Centrality')

plt.subplot(326)
closeness_centrality_list = [closeness_centrality[i] for i in range(len(closeness_centrality))]
scalar = (100/max(closeness_centrality_list))
closeness_size_list = [scalar*closeness_centrality[i] for i in range(len(closeness_centrality))]
nx.draw_networkx_nodes(G, pos,node_color=closeness_centrality_list, node_size=closeness_size_list)
nx.draw_networkx_edges(G, pos, width=0.1)
plt.title('Closeness Centrality')

#plt.show()
plt.savefig('twitter_sim_090909_centrality.png')
