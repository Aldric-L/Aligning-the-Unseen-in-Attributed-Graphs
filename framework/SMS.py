import csv
import numpy as np
import os
import gc
import igraph as ig
import matplotlib.pyplot as plt # igraph can use matplotlib for plotting
import matplotlib.cm as cm # Import colormap module


def import_dataset_fromSMS(path):
    sim_folders_dict = {}
    try:
        # List all entries in the given path
        entries = os.listdir(path)

        for entry in entries:
            full_path = os.path.join(path, entry)

            # Check if the entry is a directory and starts with 'sim_'
            if os.path.isdir(full_path) and entry.startswith('sim_'):
                # Attempt to extract the integer part after 'sim_'
                integer_str = entry[4:]
                try:
                    sim_integer = int(integer_str)
                    # Add the integer to the dictionary with a None value
                    adj = np.array(import_matrix_from_csv_numpy(f"{path}/sim_{sim_integer}/SMS-Save-OptimalGraph-{sim_integer}.csv"))
                    adj = adj[0:(adj.shape[1]), :]
                    p = extract_p_vectors_by_agentid(f"{path}/sim_{sim_integer}/SMS-Save-Vertices-{sim_integer}.csv")
                    sim_folders_dict[sim_integer] = dict(
                        adjacency_matrix=adj,
                        p_array=np.array(list(p.values()))
                    )
                except ValueError:
                    # Handle cases where the part after 'sim_' is not a valid integer
                    print(f"Warning: Folder '{entry}' does not have a valid integer after 'sim_'. Skipping.")
                    pass # Skip this folder if the integer part is invalid
                
                gc.collect()
    except FileNotFoundError:
        print(f"Error: The path '{path}' was not found.")
    except NotADirectoryError:
        print(f"Error: The path '{path}' is not a directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return sim_folders_dict

def import_matrix_from_csv_numpy(file_path):
  """
  Imports a matrix from a CSV file using NumPy.

  Args:
    file_path (str): The path to the CSV file.

  Returns:
    numpy.ndarray: A NumPy array representing the matrix,
                   or None if an error occurs.
  """
  try:
    # Use numpy.genfromtxt to load data
    # delimiter=',' specifies it's a CSV file
    # dtype=None allows numpy to infer the data type for each column
    # You could specify dtype=float if you are sure all data is numeric
    # If there are mixed types, dtype=None is better, but might result in an object array
    matrix = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding='utf-8')

    # numpy.genfromtxt might return a 1D array if there's only one row/column
    # Ensure it's at least 2D if expected
    if matrix.ndim == 1:
        # Check if it's a single row or single column and reshape accordingly
        # This assumes the CSV represents a matrix structure
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)
            num_cols = len(first_row)
        if len(matrix) % num_cols == 0: # Check if it's a flattened matrix
             matrix = matrix.reshape(-1, num_cols)
        else:
             # Handle cases that don't fit a simple reshape (e.g., single row with multiple columns)
             # If it's just a single row, genfromtxt might return 1D. Keep as is or reshape (1, num_cols)
             if len(matrix) == num_cols: # It was likely a single row
                 matrix = matrix.reshape(1, num_cols)
             # More complex 1D results might need specific handling depending on file structure

    return matrix

  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return None
  except ValueError as ve:
      print(f"Error processing data in {file_path}: {ve}")
      print("This might happen if data types are inconsistent or non-numeric entries cannot be handled.")
      return None
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None
  
def extract_p_vectors_by_agentid(csv_filepath):
    """
    Extracts 'P' vectors from a CSV file for rows where 'round' is 0,
    indexed by 'agentid'.

    Args:
        csv_filepath (str): The path to the input CSV file.

    Returns:
        dict: A dictionary where keys are agentids and values are the
              corresponding 'P' vectors (as lists of strings).
              Returns an empty dictionary if no matching rows are found
              or if the file cannot be read.
    """
    p_vectors_by_agentid = {}

    try:
        with open(csv_filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            # Strip whitespace from fieldnames
            # Create a mapping from stripped fieldname to original fieldname
            stripped_fieldnames = {field.strip(): field for field in reader.fieldnames}

            # Check if required columns exist after stripping whitespace
            required_fields = ['round', 'agentid', 'P']
            if not all(field in stripped_fieldnames for field in required_fields):
                print(f"Error: CSV file must contain the columns: {', '.join(required_fields)}. Found: {list(stripped_fieldnames.keys())}")
                return p_vectors_by_agentid # Return empty dict on error

            # Get the original field names using the stripped names
            round_fieldname = stripped_fieldnames['round']
            agentid_fieldname = stripped_fieldnames['agentid']
            p_fieldname = stripped_fieldnames['P']


            for row in reader:
                # Ensure 'round' and 'agentid' are not None and 'round' is '0'
                # Use the original field names to access row data
                round_value = row.get(round_fieldname)
                agent_id = row.get(agentid_fieldname)
                p_vector_str = row.get(p_fieldname)

                if round_value is not None and agent_id is not None and round_value.strip() == '0':
                    # Store the P vector string, indexed by agent_id
                    # Strip whitespace from agent_id just in case
                    p_vectors_by_agentid[int(agent_id.strip())] = [int(digit) for digit in p_vector_str.replace(" ", "")]

    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return p_vectors_by_agentid

def plot_graph_from_adjacency_matrix(adjacency_matrix, node_labels=None, node_color_scalars=None, cmap='viridis'):
  """
  Plots a graph from a NumPy adjacency matrix using igraph, with optional node coloring based on scalar values.

  Args:
    adjacency_matrix (numpy.ndarray): A square NumPy array representing the
                                      adjacency matrix of the graph.
                                      A[i, j] > 0 indicates an edge from node i to node j.
    node_labels (list, optional): A list of strings for node labels.
                                  If None, node indices will be used as labels.
                                  Must be the same length as the number of nodes.
    node_color_scalars (numpy.ndarray or list, optional): An array or list of scalar values,
                                                          one for each node, used to determine
                                                          the node color based on a color scale.
                                                          If None, all nodes will have the default color.
                                                          Must be the same length as the number of nodes.
    cmap (str, optional): The name of the matplotlib colormap to use for coloring nodes
                          based on node_color_scalars. Defaults to 'viridis'.
  """
  # Ensure the input is a NumPy array
  if not isinstance(adjacency_matrix, np.ndarray):
      print("Input adjacency_matrix must be a NumPy array.")
      return

  # Ensure the matrix is square
  if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
      print("Adjacency matrix must be square.")
      return

  num_nodes = adjacency_matrix.shape[0]

  # Create an igraph Graph object
  # igraph can create a graph directly from an adjacency matrix
  # mode="undirected" assumes an undirected graph. Use mode="directed" for DiGraph.
  G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist(), mode="undirected", attr="weight")

  # Set node labels
  if node_labels is None:
      # Use node indices as labels if no labels are provided
      G.vs["label"] = [str(i) for i in range(num_nodes)]
  else:
      if len(node_labels) != num_nodes:
          print(f"Error: Number of node labels ({len(node_labels)}) must match the number of nodes ({num_nodes}).")
          return
      G.vs["label"] = node_labels

  # --- Determine Node Colors ---
  vertex_colors = "lightblue" # Default color if no scalar data is provided
  if node_color_scalars is not None:
      if len(node_color_scalars) != num_nodes:
          print(f"Error: Number of node_color_scalars ({len(node_color_scalars)}) must match the number of nodes ({num_nodes}).")
          return

      # Ensure scalar data is a NumPy array for easier handling
      node_color_scalars = np.asarray(node_color_scalars)

      # Normalize the scalar values to the range [0, 1]
      # Avoid division by zero if all scalars are the same
      if np.ptp(node_color_scalars) == 0: # ptp is peak-to-peak (max - min)
          normalized_scalars = np.zeros(num_nodes)
      else:
          normalized_scalars = (node_color_scalars - np.min(node_color_scalars)) / np.ptp(node_color_scalars)

      # Get the colormap
      colormap = cm.get_cmap(cmap)

      # Map normalized scalar values to colors using the colormap
      vertex_colors = [colormap(scalar) for scalar in normalized_scalars]

  # --- Plotting the graph ---

  # igraph has its own plotting function
  # You can specify layout algorithms (e.g., "auto", "kk", "drl", "spring")
  # See igraph documentation for more layout options
  layout = G.layout("auto") # Choose an automatic layout

  # Plot the graph
  # bbox sets the size of the plot area
  # margin adds space around the plot
  # vertex_size, vertex_color, edge_width, edge_color, etc. can be customized
  fig, ax = plt.subplots() # Create a matplotlib figure and axes
  ig.plot(G,
          target=ax, # Plot onto the matplotlib axes
          layout=layout,
          bbox=(400, 400), # Bounding box for the plot
          margin=20,
          vertex_size=20,
          vertex_color=vertex_colors, # Use the determined vertex colors
          vertex_label_size=8,
          edge_width=1,
          edge_color="gray"
         )

  # Add a title (using matplotlib)
  ax.set_title("Graph from Adjacency Matrix (igraph)")

  # Display the plot
  plt.show()
