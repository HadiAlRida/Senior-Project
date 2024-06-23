import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_community
from sklearn.cluster import AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, association_rules

def build_network(results):
    G = nx.Graph()
    for result in results:
        faces = result["faces"]
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                name1 = faces[i]["Name"]
                name2 = faces[j]["Name"]
                if name1 != "unknown" and name2 != "unknown":
                    encoding1 = np.array(faces[i]["ID"])
                    encoding2 = np.array(faces[j]["ID"])
                    distance = np.linalg.norm(encoding1 - encoding2)
                    if G.has_edge(name1, name2):
                        G[name1][name2]['weight'] = min(G[name1][name2]['weight'], distance)
                    else:
                        G.add_edge(name1, name2, weight=distance)
    return G

def detect_communities_louvain(G):
    partition = community_louvain.best_partition(G, weight='weight')
    return partition

def detect_communities_girvan_newman(G):
    comp = nx_community.girvan_newman(G)
    limited = tuple(sorted(c) for c in next(comp))
    return limited

def detect_communities_agglomerative(G):
    adjacency_matrix = nx.to_numpy_array(G)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5).fit(adjacency_matrix)
    partition = {node: cluster for node, cluster in zip(G.nodes(), clustering.labels_)}
    return partition

def frequent_pattern_mining(G):
    adjacency_matrix = nx.to_numpy_array(G)
    df = pd.DataFrame(adjacency_matrix)
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    return rules

def calculate_centrality_measures(G):
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
    return {
        "degree": degree_centrality,
        "betweenness": betweenness_centrality,
        "eigenvector": eigenvector_centrality,
    }

def visualize_network(G, partition, centrality_measures):
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(partition[node]) for node in G.nodes()]
    
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300, font_size=10)
    
    plt.figure()
    plt.title('Degree Centrality')
    node_sizes = [v * 1000 for v in centrality_measures['degree'].values()]
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=node_sizes)
    
    plt.figure()
    plt.title('Betweenness Centrality')
    node_sizes = [v * 1000 for v in centrality_measures['betweenness'].values()]
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=node_sizes)
    
    plt.figure()
    plt.title('Eigenvector Centrality')
    node_sizes = [v * 1000 for v in centrality_measures['eigenvector'].values()]
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=node_sizes)

    plt.show()
