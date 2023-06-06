import math
import networkx as nx
import numpy as np

def test_graph():
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])

    G.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])
    G.add_edges_from([(2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9)])
    G.add_edges_from([(3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9)])
    G.add_edges_from([(4, 5), (4, 6), (4, 8), (4, 9)])
    G.add_edges_from([(5, 6), (5, 7), (5, 8), (5, 9)])
    G.add_edges_from([(6, 7), (6, 8), (6, 9)])

    # G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)])

    # G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)])

    # G.add_edges_from([(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)])

    return G


def calc_root(graph : nx.DiGraph):
    nodes = graph.nodes()
    max_in_degree = -1
    max_out_degree = -1
    in_set = {}
    out_set = {}
    in_list = []
    out_list = []
    in_set[0] = 0
    out_set[0] = 0
    for node in nodes:
        max_in_degree = max(max_in_degree, graph.in_degree(node))
        max_out_degree = max(max_out_degree, graph.out_degree(node))
    
        if graph.in_degree(node) in in_set:
            in_set[graph.in_degree(node)] += 1
        else:
            in_set[graph.in_degree(node)] = 1
        
        if graph.out_degree(node) in out_set:
            out_set[graph.out_degree(node)] += 1
        else:
            out_set[graph.out_degree(node)] = 1

    in_sum = 0
    out_sum = 0
    for item in in_set:
        in_sum += in_set[item]
    for item in out_set:
        out_sum += out_set[item]
    in_alpha = (in_set[0] + 1) if (in_set[0] + 1) < in_sum else (in_set[0] + 0.5)
    out_alpha = (out_set[0] + 1) if (out_set[0] + 1) < out_sum else (out_set[0] + 0.5)
    in_set[0] -= in_alpha
    out_set[0] -= out_alpha

    for i in reversed(range(0, max_in_degree + 1)):
        in_list.append(-in_set[i] if i in in_set else 0)

    for i in reversed(range(0, max_out_degree + 1)):
        out_list.append(-out_set[i] if i in out_set else 0)

    in_polynomial = np.poly1d(in_list)
    in_roots = in_polynomial.r[np.isreal(in_polynomial.r)].real
    try:
        in_root = in_roots[in_roots > 0][0]
    except:
        print("warning: subgraph with no edges")
        in_root = 1e-5
    out_polynomial = np.poly1d(out_list)
    out_roots = out_polynomial.r[np.isreal(out_polynomial.r)].real
    try:
        out_root = out_roots[out_roots > 0][0]
    except:
        out_root = 1e-5
    return in_root, out_root


def calc_difficulty(graph : nx.DiGraph):
    new_graph = graph
    if isinstance(graph, nx.Graph):
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from(graph.nodes())
        for edge in graph.edges():
            new_graph.add_edges_from([(edge[0], edge[1]), (edge[1], edge[0])])
    elif not isinstance(graph, nx.DiGraph):
        print("Unsupported graph type!")
        exit(0)
    in_root, out_root = calc_root(new_graph)
    I1 = out_root
    I2 = in_root
    I3 = (in_root + out_root ) / 2
    I4 = (math.sqrt(in_root) + math.sqrt(out_root)) / 2
    I5 = abs(math.log(in_root)) + abs(math.log(out_root))
    I7 = len(new_graph.edges()) / (math.pow(len(new_graph.nodes()), 2) - len(new_graph.nodes()))

    return I1, I2, I3, I4, I5, I7

if __name__ == '__main__':
    G = test_graph()
    print(calc_difficulty(G))
