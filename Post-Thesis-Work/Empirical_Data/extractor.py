import pickle
from DynamicModeDecomposition.subgraph_counter import subgraph_count
import collections
import re
import numpy as np
import scipy
import sys
import networkx as nx
import time
import datetime as dt
data = pickle.load(open('doge_tweet_collection_2.p', 'rb'))

def histogram(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse = True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    return deg, cnt

def convert_time_string(data, normalize=True):
    new_time_list = list()
    initial_time = data['time'][0]
    YY = 2021
    MM = 5
    DD = 28
    HH1 = int(initial_time[11:13])
    Min1 = int(initial_time[14:16])
    SS1 = int(initial_time[17:19])
    for i in range(len(data['time'])):
        if re.search(r'\b' + 'May' + r'\b', data['time'][i]):
            HH =int(data['time'][i][11:13])
            Min =int(data['time'][i][14:16])
            SS =int(data['time'][i][17:19])
            time_sec = (dt.datetime(YY,MM,DD,HH,Min,SS) - dt.datetime(YY,MM,DD,HH1,Min1,SS1)).total_seconds()
            new_time_list.append(time_sec)
    return new_time_list

def parse_text_for_retweets(data):
    texts = data['texts']
    target_accounts = []
    for text in texts:
        if text[0:4] == 'RT @':
            trgt_accnt = text.split(' ', 2)[1][1:-1]
            target_accounts.append(trgt_accnt)
        else:
            target_accounts.append(None)
    return target_accounts

def generate_adjacency_matrix(source_node_list, target_node_list):
    M = max(len(source_node_list), len(target_node_list))
    adj_matrix= np.zeros((M, M), dtype=np.int32)
    for i in range(M):
        if source_node_list[i] in target_node_list:
            if source_node_list[i] != None:
                j = [k for k, x in enumerate(target_node_list) if x == source_node_list[i]]
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    adj_matrix -= np.diag(np.diag(adj_matrix))
    adj_matrix[np.nonzero(adj_matrix)] = 1
    return adj_matrix

def generate_real_time_features_vector(data, start, N):
    max_time = int(max(data['time'])/N)
    print('The max time is %1.0f' %max_time)
    nn = max_time-start
    edge_density = np.empty(nn)
    node_count = np.empty(nn)
    edge_count = np.empty(nn)
    clustering = np.empty(nn)
    max_node_degree = np.empty(nn)

    g = nx.Graph()
    g.add_nodes_from(data['trgt_id'][:start])
    for i in range(0,start):
        if data['src_id'][i] != None:
            g.add_node(data['src_id'][i])

    for i in range(start, max_time):
        for tt in range(len(data['time'])):
            if i-1 < data['time'][tt] <= i:
                g.add_node(data['trgt_id'][tt])
                if data['src_id'][tt] != None:
                    g.add_node(data['src_id'][tt])
                    g.add_edge(u_of_edge=data['src_id'][tt], v_of_edge=data['trgt_id'][tt])
            elif data['time'][tt] > i:
                break

            mm = i - start
        # A = nx.to_numpy_matrix(g, weight=None, nonedge=0)
        #
        # A -= np.diag(np.diag(A))
        # A += A.T
        # A[np.nonzero(A)] = 1
        edge_density[mm] = nx.density(g)
        node_count[mm] = nx.number_of_nodes(g)
        edge_count[mm] = nx.number_of_edges(g)
        clustering[mm] = nx.average_clustering(g)
        max_node_degree[mm] = max([val for (node, val) in g.degree()])

        print(str(i) + ' / ' + str(max_time))
    stats = [edge_density, node_count, edge_count, clustering, max_node_degree, histogram(g)]
    return g, stats

def alt_features_vector(src, trgt, edges, start, End):
    M = int(max(src[-1][1], trgt[-1][1], edges[-1][2])/End) #this demands the list be sorted temporally
    features_vector = np.zeros((14,M-start))
    k=start
    print('The max time is %1.0f' %M)
    nn = M-k
    edge_density = np.empty(nn)
    node_count = np.empty(nn)
    edge_count = np.empty(nn)
    clustering = np.empty(nn)
    max_node_degree = np.empty(nn)
    while k<M:
        src_nodes = [src_node[0] for src_node in src if src_node[1] <= k]
        target_nodes = [trgt_node[0] for trgt_node in trgt if trgt_node[1] <= k]
        edges_list = [(edge[0], edge[1]) for edge in edges if edge[2] <= k]
        G = nx.Graph()
        G.add_nodes_from(src_nodes)
        G.add_nodes_from(target_nodes)
        G.add_edges_from(edges_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        A = nx.to_numpy_array(G)
        features_vector[:,k-start] = subgraph_count(A)
        edge_density[k-start] = nx.density(G)
        node_count[k-start] = nx.number_of_nodes(G)
        edge_count[k-start] = nx.number_of_edges(G)
        clustering[k-start] = nx.average_clustering(G)
        max_node_degree[k-start] = max([val for (node, val) in G.degree()])
        print(str(k) + ' / ' + str(M))
        stats = [edge_density, node_count, edge_count, clustering, max_node_degree, histogram(G)]
        k+=1
    return features_vector, stats, G

time_list = convert_time_string(data)
retweet_list = parse_text_for_retweets(data)
time_series = {'src_id': retweet_list, 'trgt_id': data['target_id'], 'time':time_list}
edges = list(zip(time_series['trgt_id'], time_series['src_id'], time_list))
temporal_target_nodes = list(zip(data['target_id'], time_list))
source_nodes = list(zip(retweet_list, time_list))
temporal_source_nodes = [sub for sub in source_nodes if not any(ele == None for ele in sub)]
temporal_edges = [sub for sub in edges if not any(ele == None for ele in sub)]

print(temporal_target_nodes)
print(temporal_source_nodes)
print(temporal_edges)
print(temporal_edges[-1][2])

ftmat, stats, h = alt_features_vector(temporal_source_nodes, temporal_target_nodes, temporal_edges, 8, 12)


with open('DogeCoin_2/doge_graph.p', 'wb') as fp:
    pickle.dump(h, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('DogeCoin_2/doge_graph_features.p', 'wb') as fp:
    pickle.dump(ftmat, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('DogeCoin_2/doge_graph_stats.p', 'wb') as fp:
    pickle.dump(stats, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('DogeCoin_2/cleaned_doge_data.p', 'wb') as fp:
    pickle.dump(time_series, fp, protocol=pickle.HIGHEST_PROTOCOL)