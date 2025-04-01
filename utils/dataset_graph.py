# Takes the X, y, w, u data from the cache and returns it in Pytorch Geometric format
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import knn_graph

# System
import os 
import sys

# Plotting
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection


# Aesthetic
from tqdm import tqdm

# Helper function
def compute_full_fc_edges(tc_coords):
    """
    Given an (N,3) array of 3D coordinates, return:
      - edge_index of shape (2, N*(N-1))
      - edge_attr of shape (N*(N-1), 1)
    corresponding to a fully-connected (excluding self-loops) graph.
    """
    num_nodes = tc_coords.shape[0]

    # Compute the NxN distance matrix
    dist_mat = cdist(tc_coords, tc_coords, metric="euclidean")  # shape: (N, N)

    # We want all edges except self-loops => mask out the diagonal
    mask = np.ones((num_nodes, num_nodes), dtype=bool)
    np.fill_diagonal(mask, False)  # remove self-loops

    # edge_attr: flatten out the distances where mask==True
    edge_attr = dist_mat[mask]  # shape: (N*(N-1),)
    edge_attr = edge_attr[:, None]  # make it (N*(N-1), 1)

    # edge_index: the (row, col) pairs
    row_idx, col_idx = np.where(mask)
    # stack into shape (2, N*(N-1))
    edge_index = np.stack([row_idx, col_idx], axis=0)

    return edge_index, edge_attr

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, w, u, use_knn=False, k=5):
        self.data_list = []
        self.fields = X.columns.values
        self.use_knn = use_knn
        self.k = k

        # Precompute the data objects per cluster
        print("Creating data objects")
        if use_knn:
            print(f"Using kNN with k={k}")
        else:
            print("Using fully connected graph")
        for idx in tqdm(range(len(y))):
            #if idx >= 10: 
            #    print("Only printing the first 10 clusters")
            #    break
            nodes = X.xs(idx, level="entry")
            tc_coords = nodes.loc[:, ["x", "y", "z"]].values
            x = torch.tensor(nodes.values, dtype=torch.float)

            num_nodes = x.size(0)
            num_edges = (num_nodes*num_nodes) - num_nodes # Fully connected graph
            num_edge_features = 1 # Only the distance between nodes
            
            # For loop version
            """
            edge_attr = np.zeros((num_edges, num_edge_features), dtype=float)
            edge_index = np.zeros((2, num_edges), dtype=int)
            i_edge = 0
            for i_node in range(num_nodes):
                for j_node in range(num_nodes):
                    if i_node == j_node: continue

                    # Compute the distance between the nodes
                    dist_ij = np.linalg.norm(tc_coords[i_node] - tc_coords[j_node])
                    edge_attr[i_edge, 0] = dist_ij
                    edge_index[0, i_edge] = i_node
                    edge_index[1, i_edge] = j_node
                    i_edge += 1
            """

            # Vectorized version
            if use_knn:
                pos = torch.tensor(tc_coords, dtype=torch.float)
                edge_index = knn_graph(pos, k=k, loop=False)
                # Compute edge attributes (Euclidean distances) from the knn graph
                src = pos[edge_index[0]]
                dst = pos[edge_index[1]]
                edge_attr = torch.norm(src - dst, dim=1, p=2).unsqueeze(1)

            else:    
                edge_index, edge_attr = compute_full_fc_edges(tc_coords)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long)

            y_idx = torch.tensor(y.loc[idx].values, dtype=torch.float)
            w_idx = torch.tensor(w.loc[idx].values, dtype=torch.float)
            u_idx = torch.tensor(u.loc[idx].values, dtype=torch.float)
            if u_idx.dim() == 1:
                u_idx = u_idx.unsqueeze(0)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_idx, w=w_idx, u=u_idx)

            #print(f"Data object for cluster {idx}: {data}")

            self.data_list.append(data)
            # Debug print
            #print(f"Cluster {idx} has {len(nodes)} trigger cells (nodes)")
            #print(tc_coords)
            #print(x)


    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    # Function to draw a data object as a cluster
    def plot_data(self, data_index=0, draw_layer=1, plot_dir="graphs"):
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        print(f"Plotting data object {data_index} for layer {draw_layer} as a graph")
        data = self.data_list[data_index]
        fields = self.fields.tolist()
        # Find the fields corresponding to positions
        x_idx = fields.index("x")
        y_idx = fields.index("y")
        z_idx = fields.index("z")
        layer_idx = fields.index("layer")
        G = nx.Graph()

        # Add the nodes
        for i_node in range(data.x.size(0)):
            node_features = data.x[i_node].numpy()  
            node_pos = node_features[[x_idx, y_idx, z_idx]]
            node_layer_val = node_features[layer_idx]
            if node_layer_val != draw_layer: continue
            G.add_node(i_node, pos=(node_pos[0], node_pos[1]))

        # Add the edges
        for i_edge in range(data.edge_index.size(1)):
            edge = data.edge_index[:, i_edge].numpy()
            G.add_edge(edge[0], edge[1])

        # Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True)
        plt.savefig(f"{plot_dir}/graph_{data_index}.png")
        plt.clf()

        return G
    
    def plot_data_3d(self, data_index, plot_dir="graphs", draw_edges=False):
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        data = self.data_list[data_index]
        fields = self.fields.tolist()
        
        # Find the fields corresponding to positions
        x_idx = fields.index("x")  # old x
        y_idx = fields.index("y")  # old y
        z_idx = fields.index("z")  # old z
        
        # Grab the original coordinates: shape (num_nodes, 3) => [old_x, old_y, old_z]
        tc_coords = data.x[:, [x_idx, y_idx, z_idx]].numpy()
        
        # Re-map so that old_x -> new_z, old_y -> new_x, old_z -> new_y
        # i.e. [new_x, new_y, new_z] = [old_y, old_z, old_x]
        new_coords = tc_coords.copy()
        new_coords[:, 0] = tc_coords[:, 1]  # new X = old Y
        new_coords[:, 1] = tc_coords[:, 2]  # new Y = old Z
        new_coords[:, 2] = tc_coords[:, 0]  # new Z = old X
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes using the swapped coordinates
        ax.scatter(
            new_coords[:, 0],  # new X
            new_coords[:, 1],  # new Y
            new_coords[:, 2],  # new Z
            c='b', s=20, label='Nodes'
        )
        
        if draw_edges:
            # For each edge, draw a line between the corresponding nodes in the new coordinate system
            for i_edge in range(data.edge_index.size(1)):
                start, end = data.edge_index[:, i_edge].tolist()
                x_vals = [new_coords[start, 0], new_coords[end, 0]]
                y_vals = [new_coords[start, 1], new_coords[end, 1]]
                z_vals = [new_coords[start, 2], new_coords[end, 2]]
                ax.plot(x_vals, y_vals, z_vals, c='gray', linewidth=0.5)

        # Get cluster (global pT eta and phi for title)
        cluster_kinematics = (data.u.numpy())[0]
        print(f"Cluster kinematics: {cluster_kinematics}")
        
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_zlabel('X')
        ax.set_title(f"3D TC Graph for cluster {data_index}\n pT: {cluster_kinematics[0]:.2f} GeV, eta: {cluster_kinematics[1]:.2f}, phi: {cluster_kinematics[2]:.2f}")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/graph3d_{data_index}.png")
        plt.close(fig)
