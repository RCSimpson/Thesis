import numpy
import pickle
import matplotlib.pyplot as plt
from DynamicModeDecomposition.plotter import data_plot
import networkx as nx

data = pickle.load(open('DogeCoin_2/doge_graph_features.p', 'rb'))

print(data[:,100])
print(data[:,200])
print(data[:,data.shape[1]-1])
data_plot(data= data,
          name='Dogecoin_2/doge_features_plot',
          show=False,
          save=True)

h = pickle.load(open('DogeCoin_2/doge_graph.p', 'rb'))
fig = plt.figure(figsize=(10,5))
plt.subplot(121)
pos=nx.spring_layout(h, k=0.2, iterations=20)
#pos = nx.random_layout(h)
nx.draw_networkx_nodes(h, pos, node_size=4)
nx.draw_networkx_edges(h, pos, arrowsize=4, alpha=0.2)
plt.subplot(122)
lcc = max(nx.connected_components(h), key=len)
nx.draw_networkx_nodes(h, nodelist=lcc, pos=pos, node_size=4)
nx.draw_networkx_edges(h, nodelist=lcc, edgelist=h.edges(lcc), pos=pos, arrowsize=4, alpha=0.2)
plt.savefig('Dogecoin_2/doge_graph.png')

stats = pickle.load(open('DogeCoin_2/doge_graph_stats.p', 'rb'))

fig = plt.figure(figsize=(10,10))
plt.subplot(321)
plt.plot(stats[1], label='Node Count')
plt.plot(stats[2], label='Edge Count')
plt.legend()

plt.subplot(322)
plt.plot(stats[0], label='Edge Density')
plt.legend()

plt.subplot(323)
plt.plot(stats[3], label='Avg Clustering')
plt.legend()

plt.subplot(324)
plt.plot(stats[4], label='Greatest Degree')
plt.legend()

plt.subplot(325)
plt.scatter(stats[5][0],stats[5][1])
plt.savefig('DogeCoin_2/doge_graph_stats.png')
