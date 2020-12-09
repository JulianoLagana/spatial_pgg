import networkx as nx

def read_file_net(filename):    
    '''
    Function designed to read a network from file and generate a `graph` object out of it.
    It is thought to work in the list of edges format, but may work in others compatible with
    networkx
    '''
    edges = []
     
    for line in open(filename):
        line = line.strip().split()
        edge = [int(value) for value in line]
        edges.append(edge)
    
    G = nx.Graph(edges)
    n_nodes = G.number_of_nodes()
    
    return G, n_nodes
