import csv
import logging
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_community
from sklearn.cluster import AgglomerativeClustering
from mlxtend.frequent_patterns import apriori, association_rules
from pyvis.network import Network
import networkx as nx
import cv
from database_handler import create_connection, execute_query, return_create_statement_from_df, return_insert_into_sql_statement_from_df

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
    
    # Create a dictionary mapping nodes to their communities
    partition = {}
    for community_id, community in enumerate(limited):
        for node in community:
            partition[node] = community_id
            
    return partition


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
    eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight',max_iter=1000)
    return {
        "degree": degree_centrality,
        "betweenness": betweenness_centrality,
        "eigenvector": eigenvector_centrality,
    }


def visualize_network(G, partition, centrality_measures):
    net = Network(notebook=True)
    
    # Add nodes with partition as group
    for node in G.nodes():
        net.add_node(node, group=partition[node], title=f"Degree: {centrality_measures['degree'][node]:.2f}Betweenness: {centrality_measures['betweenness'][node]:.2f}Eigenvector: {centrality_measures['eigenvector'][node]:.2f}")
    
    # Add edges with weight
    for source, target, data in G.edges(data=True):
        net.add_edge(source, target, value=data['weight'])
    
    # Generate network layout and show
    net.show("network.html")


def export_friendship_data_to_db(G, schema_name, table_name, detailed_table_name, partition, centrality_measures, filename="friendship_data.csv"):
    # Export friendship network data with weights
    edges = [{'Source': u, 'Target': v, 'Weight': d['weight']} for u, v, d in G.edges(data=True)]
    df_edges = pd.DataFrame(edges)
    print(df_edges)
    
    db_session = create_connection()
    try:
        # Create and insert into edges table
        create_edges_table_statement = return_create_statement_from_df(df_edges, schema_name, table_name)
        execute_query(db_session, create_edges_table_statement)
        
        insert_edges_statements = return_insert_into_sql_statement_from_df(df_edges, schema_name, table_name)
        for insert_statement in insert_edges_statements:
            execute_query(db_session, insert_statement)
        
        # Export detailed friendship data
        detailed_data = []
        nodes = list(G.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                person1 = nodes[i]
                person2 = nodes[j]
                
                relationship = "friends" if G.has_edge(person1, person2) else "not friends"
                degree_centrality1 = centrality_measures['degree'].get(person1, 0)
                degree_centrality2 = centrality_measures['degree'].get(person2, 0)
                same_community = partition.get(person1) == partition.get(person2)
                
                detailed_data.append({
                    'Person1': person1,
                    'Person2': person2,
                    'Relationship': relationship,
                    'DegreeCentrality1': degree_centrality1,
                    'DegreeCentrality2': degree_centrality2,
                    'SameCommunity': same_community
                })
        
        df_detailed = pd.DataFrame(detailed_data)
        print(df_detailed)
        
        # Create and insert into detailed data table
        create_detailed_table_statement = return_create_statement_from_df(df_detailed, schema_name, detailed_table_name)
        execute_query(db_session, create_detailed_table_statement)
        
        insert_detailed_statements = return_insert_into_sql_statement_from_df(df_detailed, schema_name, detailed_table_name)
        for insert_statement in insert_detailed_statements:
            execute_query(db_session, insert_statement)
    except Exception as e:
        logging.error(f"Error exporting friendship data to database: {e}")
    finally:
        db_session.close()
