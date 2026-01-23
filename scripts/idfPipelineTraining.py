import numpy as np
import torch 
import networkx as nx
from framework.trainFct import *
from torch_geometric.data import Data
from framework.visuals import *
import copy
import os
from framework.boundedManifold import BoundedManifold
from framework.distanceApproximations import DistanceApproximations
from framework.KLAnnealingScheduler import NoKLScheduler

if __name__ == '__main__':
    datasetPath = "data/France/transportsIDF/"

    def read_matrix_from_csv_loadtxt(filepath, delimiter=',', dtype=float):
        """
        Reads a NumPy matrix from a CSV file using np.loadtxt().

        Args:
            filepath (str): The path to the CSV file.
            delimiter (str): The character separating values in the CSV file (default is comma).

        Returns:
            numpy.ndarray: The matrix read from the CSV file.
        """
        try:
            matrix = np.loadtxt(filepath, delimiter=delimiter, dtype=dtype)
            print(f"Successfully loaded matrix from {filepath} using np.loadtxt().")
            return matrix
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return None

    p_vectors_array = read_matrix_from_csv_loadtxt(datasetPath + "idf_attr_nnorm.csv")
    adjacency_matrix = read_matrix_from_csv_loadtxt(datasetPath + "idf_adjacency_nnorm.csv")
    names = read_matrix_from_csv_loadtxt(datasetPath + "idf_names_nnorm.csv", dtype=str)
    num_nodes = adjacency_matrix.shape[0]
    dimP = 20

    G = nx.from_numpy_array(adjacency_matrix)
    G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops
    latent_dim = 2
    input_dim = dimP
    batch_size = 256


    dataset = []
    data = Data(x=torch.tensor(p_vectors_array, dtype=torch.float), 
                edge_index=adj_matrix_to_edge_index(adjacency_matrix)[0], 
                edge_labels=adj_matrix_to_edge_index(adjacency_matrix)[1],
                adjacency_matrix=torch.tensor(adjacency_matrix))
    dataset.append(data)

    # Select a single graph to train on
    single_graph = dataset[0]

    # Wrap in list for compatibility with DataLoader-like expectations
    single_graph_list = [single_graph]
    dropout = 0
    phase1_epochs = 1200
    #phase1_epochs = 300
    phase2_epochs = 200
    #phase2_epochs = 100
    lr_phase1 = 0.005
    latent_dim = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"


    encoder = MLPEncoder(
        input_dim=input_dim,
        hidden_dims=[96, 64],
        latent_dim=latent_dim,
        mlp_layers=2,
        dropout=0.2,
        activation=nn.ReLU()
    )

    node_decoder = NodeAttributeVariationalDecoder(
        latent_dim=latent_dim,
        output_dim=input_dim,
        #hidden_dims=[5000, 128],
        #hidden_dims=[2000, 128],
        hidden_dims=[128],
        dropout=dropout,
        activation=nn.ReLU(),
        loss_options= {
            "lambda_comp_variance": 100,
            "lambda_decoder_variance":20,
            "debug": False},
        clip_var = 30,
    )

    # Create KL annealing scheduler
    kl_scheduler = KLAnnealingScheduler(
        anneal_start=0.0,
        #anneal_end=0.001,
        #anneal_end=0.8,
        anneal_end=1,
        anneal_steps=phase1_epochs * len(single_graph_list),
        anneal_type='sigmoid',
    )

    # Create initial model with only node decoder
    model_phase1 = GraphVAE(
        encoder=encoder,
        decoders=[node_decoder],
        kl_scheduler=kl_scheduler,
        compute_latent_manifold=False,
    )

    if os.path.exists("model_phase1_IDF.pth"):
        print("Loading pretrained model")
        model_phase1.load_state_dict(torch.load('model_phase1_IDF.pth'))
    else:
        print("=== Starting Phase 1: Training encoder with node feature reconstruction ===")

        # Phase 1 training
        history_phase1 = train_phase1(
            model=model_phase1,
            data_loader=single_graph_list,
            num_epochs=phase1_epochs,
            lr=lr_phase1,
            weight_decay=1e-5,
            verbose=True,
            device=device,
            loss_coefficient=1
        )

        print("\n=== Phase 1 Complete ===")

        torch.save(model_phase1.state_dict(), 'model_phase1_IDF.pth')
        print("\n=== Phase 1 Saved ===")

        visualize_training(history_phase1)
        visualize_node_features_reconstruction(model_phase1, single_graph, sample_features=dimP)
        visualize_latent_space(model_phase1, [single_graph])

    
    model_phase1 = model_phase1.to('cpu')
    model_phase2 = copy.deepcopy(model_phase1)

    model_phase1.eval()

    with torch.no_grad():
        x = single_graph.x.to(device)
        edge_index = single_graph.edge_index.to(device)
        latent_mu = model_phase1.encode(x, edge_index=edge_index)

    latent_points = latent_mu[0]

    lr_phase2 = 0.001
    distance_mode = "dijkstra" # "linear_interpolation"

    print("=== Starting Phase 2: Freezing encoder and adding adjacency decoder ===")

    model_phase2.set_compute_latent_manifold(True)
    model_phase2.construct_latent_manifold(bounds=BoundedManifold.hypercube_bounds(latent_points, margin=0.1, relative=True), force=True)
    model_phase2.set_encoder_freeze(True)

    #torch.autograd.set_detect_anomaly(True)

    distance_decoder = ManifoldHeatKernelDecoder(
        distance_mode=distance_mode,
        latent_dim=latent_dim,
        #heat_time=torch.logspace(start=-1, end=2, steps=10).tolist(),  # Adjust for local vs global structure
        num_eigenvalues=500,
        num_integration_points=20,
        name="adj_decoder",
        max_ema_epochs=phase2_epochs,
        #ema_lag_factor=0.1
    )

    # Add to your GraphVAE model
    model_phase2.add_decoder(distance_decoder)

    # Set reference decoder (the node attribute decoder)
    #model_phase2.get_decoder("adj_decoder").giveManifoldInstance(model_phase2.get_latent_manifold())
    model_phase2.get_decoder("adj_decoder").giveVAEInstance(model_phase2)

    # Reset KL scheduler for phase 2
    model_phase2.kl_scheduler = NoKLScheduler()

    torch.nn.utils.clip_grad_norm_(model_phase2.parameters(), 1)
    #torch.autograd.set_detect_anomaly(True)

    # Phase 2 training
    history_phase2 = train_phase2(
        model=model_phase2,
        data_loader=single_graph_list,
        latent_points=latent_points,
        num_epochs=phase2_epochs,
        lr=lr_phase2,
        weight_decay=1e-5,
    #    decoder_weights={"adj_decoder": -1, "node_attr_decoder":-1 },
        decoder_weights={"adj_decoder": 1, "node_attr_decoder":0.0 },
        verbose=True,
        device=device,
    )

    print("\n=== Phase 2 Complete ===")
    torch.save(model_phase2.state_dict(), 'model_phase2_idf_dijkstra.pth')