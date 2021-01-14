import warnings
from pprint import pprint
from mpl_toolkits import mplot3d
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import pickle
import re
from PIL import Image

def subgraph_count(G):

    A = nx.adjacency_matrix(G)
    Af = A.todense()
    fvec = np.zeros(14)
    nds, nds = Af.shape
    dgs = np.zeros((nds, 1), dtype=np.int)
    ovec = np.ones((nds, 1), dtype=np.int)
    dfvec = np.zeros((nds, 1), dtype=np.int)
    dftvec = np.zeros((nds, 1), dtype=np.int)
    dgs[:, 0] = np.sum(Af, 1).flatten()
    dfvec[:, 0] = dgs[:, 0] - ovec[:, 0]
    dftvec[:, 0] = dgs[:, 0] - 2 * ovec[:, 0]

    dgs[:] = np.array([np.sum(Af, 1)])
    ovec[:] = np.array([np.ones(dgs.size)]).T
    dfvec[:] = dgs[:] - ovec[:]
    dftvec[:] = dgs[:] - 2. * ovec[:]

    A = nx.adjacency_matrix(G)
    Af = A.todense()
    A2 = Af @ Af
    A2f = Af * A2
    # Q = np.multiply(A2,A)
    A3 = Af @ A2
    A3d = np.diag(A3)
    A4 = Af @ A3
    A5 = Af @ A4
    P1 = np.sum(dgs) / 2
    P2 = np.sum(dgs * dfvec) / 2.
    A2m1 = A2 - np.ones(A2.shape)
    a2chs = .5 * A2 * A2m1
    a2chsvec = np.sum(a2chs - np.diag(np.diag(a2chs)), 1)

    # note, first three entries of fvec are the number of cycles C_3, C_4, and C_5

    fvec[0] = np.trace(A3) / 6.  # n_G(C_3)
    fvec[1] = (np.trace(A4) - 4. * P2 - 2. * P1) / 8.  # n_G(C_4)
    fvec[3] = .5 * np.sum(np.sum(Af * (dfvec @ dfvec.T))) - 3. * fvec[0]  # n_G(H_3)
    fvec[4] = np.sum(dgs[:, 0] * dfvec[:, 0] * dftvec[:, 0]) / 6.  # n_G(H_4)
    fvec[5] = np.sum(A3d * dftvec[:, 0]) / 2.  # n_G(H_5)
    fvec[2] = (np.trace(A5) - 10 * fvec[5] - 30 * fvec[0]) / 10.  # n_G(C_5)

    fvec[6] = 1 / 4. * np.sum(np.sum(A2f * A2m1))  # n_G(H_6)
    fvec[7] = 1 / 4. * np.dot(A3d, (dftvec[:, 0] * (dftvec[:, 0] - np.ones(nds))))  # n_G(H_7)
    fvec[8] = 1 / 2. * (dftvec.T @ (A2f @ dftvec)) - 2. * fvec[6]  # n_G(H_8)
    fvec[9] = np.dot(dftvec[:, 0], a2chsvec) - 2. * fvec[6]  # n_G(H_9)
    fvec[10] = .5 * np.dot(A3d, np.sum(A2 - np.diag(np.diag(A2)), 1)) - 6. * fvec[0] - 2. * fvec[5] - 4. * fvec[
        6]  # n_G(H_10)
    fvec[11] = .5 * np.sum(.5 * A3d * (.5 * A3d - 1.)) - 2. * fvec[6]  # n_G(H_11)
    fvec[12] = .5 * np.sum(np.sum(A2f * A3)) - 9. * fvec[0] - 2. * fvec[5] - 4. * fvec[6]  # n_G(H_12)
    fvec[13] = 1. / 12. * np.sum(np.sum(A2f * (A2 - np.ones(A2.shape)) * (A2 - 2. * np.ones(A2.shape))))  # n_G(H_13)

    return fvec

def graph_visualizer(G):
    plt.subplot(211)
    options = {'node_color': 'black','node_size': 10,'width': 1}
    nx.draw(G,**options)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.subplot(212)
    dgs = np.array([d for n, d in G.degree()])
    pdist = dgs/np.sum(dgs)
    lgp = np.ma.log10(pdist)
    lgp = np.sort(lgp)
    plt.plot(np.log10(np.arange(pdist.size)+1),lgp[::-1])

# probabilities
l = 0.5
p = 0.5
p_t1 = l / (l + 1)
p_t2 = p / (l + 1)
p_t3 = (1 - p) / (l + 1)
q = 0.9
p_superstar = q
p_nonsuperstar = 1 - q


def generate_arrival(node, G, node_dict, p_t1, p_t2, p_t3, p_superstar, p_nonsuperstar):
    # generate random float between 0 and 1
    arrival = np.random.choice(['t1', 't2', 't3'], p=[p_t1, p_t2, p_t3])
    # T1 arrival
    if arrival == 't1':
        new_node = node + 1
        G.add_node(new_node)
        node_dict['t1'].append(new_node)
    # T2 arrival
    elif arrival == 't2':
        new_node = node + 1
        tree_node = generate_tree(G, node_dict)
        attach_node = generate_tree_node(
            tree_node, G, p_superstar, p_nonsuperstar)
        G.add_edge(attach_node, new_node)
        node_dict['t2'].append(new_node)
    # T3 arrival
    else:
        tree_node = generate_tree(G, node_dict)
        attach_node = generate_tree_node(
            tree_node, G, p_superstar, p_nonsuperstar)
        old_node = generate_t3_node(attach_node, G)
        G.add_edge(attach_node, old_node)
        node_dict['t3'].append(old_node)
        new_node = node
    # plt.clf()
    # nx.draw_networkx(G, node_size=100)
    return new_node, G, node_dict


# probability attachment to a tree (subgraph)


def generate_tree(G, node_dict):
    p_tree_dict = {}
    t1_nodes = node_dict['t1']
    total = 0
    for node in t1_nodes:
        # tree = nx.descendants(G, node)
        tree = G.out_degree(node)
        # tree_length = len(tree) + 1
        tree_length = tree + 1
        total += tree_length
    for node in t1_nodes:
        # tree = nx.descendants(G, node)
        tree = G.out_degree(node)
        # p_tree = (len(tree) + 1)/total
        p_tree = (tree + 1) / total
        p_tree_dict[node] = p_tree
    keys = list(p_tree_dict.keys())
    values = list(p_tree_dict.values())
    tree_node = np.random.choice(keys, p=values)
    return tree_node


# probability attachment to a node within tree (t2 and t3)
# choose superstar node


def generate_tree_node(tree_node, G, p_superstar=p_superstar, p_nonsuperstar=p_nonsuperstar):
    superstar = tree_node
    p_node_dict = {}
    p_node_dict[superstar] = p_superstar
    # tree = nx.descendants(G, tree_node)
    tree = list(G.successors(tree_node))
    if len(tree) > 0:
        non_superstars = tree
        total = 0
        for node in non_superstars:
            # weighted_node = len(nx.descendants(G, node)) + 1
            weighted_node = len(list(G.successors(node))) + 1
            total += weighted_node
        for node in non_superstars:
            # weighted_node = len(nx.descendants(G, node)) + 1
            weighted_node = len(list(G.successors(node))) + 1
            p_node_dict[node] = p_nonsuperstar * (weighted_node / total)
        keys = list(p_node_dict.keys())
        values = list(p_node_dict.values())
        node = np.random.choice(keys, p=values)
    else:
        node = superstar
    return node


# probability existing node (t3)


def generate_t3_node(node, G):
    nodes = list(G.nodes)
    nodes.remove(node)
    node = np.random.choice(nodes)
    return node



def model_simulation(t_max, G=None, node_dict=None, l=0, p=0):
    # parameters
    g_densities = []
    lcc_densities = []
    lcc_nodes = []
    lcc_edges = []
    times = []
    l_values = []
    p_values = []

    indhg = np.int(3 * t_max / 4)
    indlw = np.int(t_max / 16.)
    ftmat = np.zeros((14, indhg - indlw + 1))
    dynamic_edge_density = np.zeros(indhg - indlw + 1)
    node_count = np.zeros( indhg - indlw + 1)
    edge_count = np.zeros( indhg - indlw + 1)
    largest_connected_node_degree = np.zeros( indhg - indlw + 1)

    # initialize graph
    G = nx.DiGraph()
    node_dict = {'t1': [], 't2': [], 't3': []}
    node = 0
    G.add_node(node)
    node_dict['t1'].append(node)
    # parameters
    l = l
    p = p
    q = 0.9
    p_superstar = q
    p_nonsuperstar = 1 - q

    # simulation
    t = 1
    tcnt = 0

    while t <= t_max:
        # calculate parameters
        p_t1 = l / (l + 1)
        p_t2 = p / (l + 1)
        p_t3 = (1 - p) / (l + 1)
        node, G, node_dict = generate_arrival(node=node, G=G, node_dict=node_dict, p_t1=p_t1,
                                              p_t2=p_t2, p_t3=p_t3, p_superstar=p_superstar,
                                              p_nonsuperstar=p_nonsuperstar)

            # extract features
        if t < indhg and t >= indlw:

            H = G.to_undirected()
            ftmat[:, tcnt] = subgraph_count(H)
            # dynamic_edge_density[tcnt] = nx.density(H)
            # node_count[tcnt] = nx.number_of_nodes(H)
            # edge_count[tcnt] = nx.number_of_edges(H)
            # largest_connected_node_degree[tcnt] = len(nx.degree_histogram(H))

            # We will use these specific graphs to later to graph the largest connected component
            # if tcnt == 10:
            #     G10 = H
            # if tcnt == 50:
            #     G50 = H
            # if tcnt == 100:
            #     G100 = H
            # if tcnt == 200:
            #     G200 = H
            # if tcnt == 687:
            #     GG = H

            tcnt += 1
        t += 1

    # descriptor = np.zeros((4,indhg - indlw + 1))
    # descriptor[0,:] = dynamic_edge_density
    # descriptor[1, :] = node_count
    # descriptor[2, :] = edge_count
    # descriptor[3, :] = largest_connected_node_degree

    # file_name = 'feature_data' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'
    # file_name_1 = 'graph_data' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'
    # file_name_2 = 'graph_data_10' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'
    # file_name_3 = 'graph_data_50' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'
    # file_name_4 = 'graph_data_100' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'
    # file_name_5 = 'graph_data_200' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'
    # file_name_6 = 'descriptor' + '_l=' + str(l) + '_q=' + str(q) + '_p=' + str(p) + '.p'

    #pickle.dump(ftmat, open(file_name, "wb"))
    #pickle.dump(GG, open(file_name_1, "wb"))
    #pickle.dump(G10, open(file_name_2, "wb"))
    #pickle.dump(G50, open(file_name_3, "wb"))
    #pickle.dump(G100, open(file_name_4, "wb"))
    #pickle.dump(G200, open(file_name_5, "wb"))
    #pickle.dump(descriptor, open(file_name_6, "wb"))

    return ftmat

## Need to modify the simulation so that it does most of the data prep for me and with a wider arraw of parameter choices.


t_max = 500
l = 0.6
p = 0.6
p_t1 = l / (l + 1)
p_t2 = p / (l + 1)
p_t3 = (1 - p) / (l + 1)
q = 0.9
p_superstar = q
p_nonsuperstar = 1 - q

ftmat = model_simulation(t_max, G=None, node_dict=None, l=l, p=p)

# NN=300
# data_tot = np.empty((1, 9, NN))
# tot_params = np.empty((1, 3))
#
# for k in np.arange(0.2, 1,0.2):
#     for j in np.arange(0.2,1,0.2):
#             i=0
#
#             while True:
#                 exception_1 = False
#                 exception_2 = False
#
#                 try:
#                     l = k
#                     p = j
#                     ftmat = model_simulation(t_max, G=None, node_dict=None, l=l, p=p)
#                     #print(ftmat.shape)
#                     print(ftmat.shape)
#                 except:
#                     print('exception 1 triggered')
#                     exception_1 = True
#
#                 if exception_1 == False:
#                     try:
#                         #A = nx.to_pandas_adjacency(G, dtype=int)
#                         #A = A.to_numpy()
#                         file_name = str(l) + "_" + str(p) + "_" + str(q)  + "_" + str(i)
#                         # pickle.dump(G, open(file_name+".p", "wb"))
#
#                         ftmat = ftmat[:9,:NN]
#                         data = ftmat.reshape(1, 9, NN)
#
#                         data_tot = np.concatenate((data_tot, data), axis=0)
#                         params = np.array([[l,p,q]])
#                         params = params.reshape((1,3))
#                         tot_params = np.concatenate((tot_params, params), axis = 0)
#                         i+=1
#                         print(i)
#                         if i ==10:
#                             break
#                     except:
#                         #print('G was not calculated because of the random seed')
#                         print('exception 2 triggered')
#                         pass
#
#
# data_with_params = {"features": data_tot, "params": tot_params}
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle

with open('../Data/features_matrix_sample.p', 'wb') as fp:
    pickle.dump(ftmat, fp, protocol=pickle.HIGHEST_PROTOCOL)
        #file_name = str(l) + "_" + str(p)+ "_" + str(q)# + str(i)
#pickle.dump(G, open(file_name, "wb"))
