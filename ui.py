import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from neural_network import NeuralNetwork, sigmoid
import networkx as nx

# Function to visualize neural network architecture
def custom_layout(nn):
    # Compute the positions of the nodes
    pos = {}
    layer_width = 1.0 / (len(nn.layers) - 1)  # Width of each layer in the layout
    for i, layer in enumerate(nn.layers):
        num_nodes = layer.weights.shape[0]
        node_height = 1.0 / num_nodes  # Height of each node in the layout
        for j in range(num_nodes):
            pos[(i, j)] = (i * layer_width, j * node_height)  # Position of the node in the layout
    return pos

def visualize_nn_architecture(nn):
    # Initialize the neural network with a forward pass of random data
    #nn.forward_pass(np.random.rand(nn.layers[0].input_size, 1))

    G = nx.DiGraph()

    # Add neurons as nodes with activations as node attributes
    for i, layer in enumerate(nn.layers):
        for j in range(layer.weights.shape[0]):
            activation = layer.activation[j][0] if layer.activation is not None else 0.0
            G.add_node((i, j), activation=activation)

    # Add edges between neurons representing weights
    for i in range(len(nn.layers) - 1):
        for j in range(nn.layers[i + 1].weights.shape[0]):
            for k in range(nn.layers[i + 1].weights.shape[1]):
                weight = nn.layers[i + 1].weights[j][k]
                G.add_edge((i, k), (i + 1, j), weight=weight)

    # Draw the graph
    pos = custom_layout(nn)  # Use custom layout function
    node_labels = {(i, j): f"{G.nodes[(i, j)]['activation']:.2f}" for i, j in G.nodes()}
    edge_labels = {(i, j): f"{G.edges[(i, j)]['weight']:.2f}" for i, j in G.edges()}
    edge_width = [abs(G.edges[edge]['weight']) for edge in G.edges()]

    fig, ax = plt.subplots()
    nx.draw(G, pos, ax=ax, with_labels=False, node_size=500, node_color='skyblue', edge_color='black', width=edge_width, arrows=True)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    ax.set_title('Neural Network Architecture')
    return fig


# Function to perform training animation
def train_nn_animation(nn, training_data, epochs, mini_batch_size, learning_rate):
    n = len(training_data)
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            nn.train(mini_batch, learning_rate)

        # Update visualization after each epoch
        fig = visualize_nn_architecture(nn)
        st.pyplot(fig)

# Create a Streamlit app
st.title('Neural Network Training Animation')

# Initialize neural network and input data
nn = NeuralNetwork(shape=(5, 8, 6, 4, 3))  # Example architecture (5 input, 3 hidden, 2 output neurons)

training_data = [(np.random.rand(5, 1), np.random.rand(3, 1)) for _ in range(20)]  # Generate 20 random training examples

epochs = st.number_input('Number of epochs', min_value=1, max_value=100, value=5)
mini_batch_size = st.number_input('Mini batch size', min_value=1, max_value=20, value=5)
learning_rate = st.number_input('Learning rate', min_value=0.01, max_value=1.0, value=0.1)

# Create a button to visualize initial architecture
if st.button('Show Initial Architecture'):
    fig = visualize_nn_architecture(nn)
    st.pyplot(fig)

# Create a button to perform training animation
if st.button('Start Training'):
    train_nn_animation(nn, training_data, epochs, mini_batch_size, learning_rate)
