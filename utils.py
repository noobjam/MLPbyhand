import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import streamlit as st


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# def train_neural_network(nn, training_data):
#     nn.train(training_data, epochs=100, mini_batch_size=10, eta=0.1)
#     print("Training complete!")


def visualize_neural_network(nn):
    # Visualize the neural network architecture
    shape = nn.shape
    num_layers = len(shape)
    
    fig = go.Figure()
    
    # Add nodes for each layer
    for i in range(num_layers):
        layer_nodes = [f'Layer {i+1} Node {j+1}' for j in range(shape[i])]
        fig.add_trace(go.Scatter(x=[i]*shape[i], y=layer_nodes, mode='markers',
                                 marker=dict(size=20, color=i, colorscale='Viridis'),
                                 name=f'Layer {i+1}'))
    
    # Add edges between nodes
    for i in range(num_layers - 1):
        layer_nodes_current = [f'Layer {i+1} Node {j+1}' for j in range(shape[i])]
        layer_nodes_next = [f'Layer {i+2} Node {j+1}' for j in range(shape[i+1])]
        
        for current_node in layer_nodes_current:
            for next_node in layer_nodes_next:
                fig.add_trace(go.Scatter(x=[i, i+1], y=[current_node, next_node], mode='lines',
                                         line=dict(color='black', width=1), showlegend=False))
    
    fig.update_layout(title='Neural Network Architecture', xaxis=dict(title='Layer'), yaxis=dict(title='Node'))
    st.plotly_chart(fig)

def visualize_forward_pass(ax, nn,training_data):
    # Visualize the forward pass calculations at each layer
    input_data = training_data[0][0]  # Take the first input data from the training data
    activations, _ = nn.forward_pass(input_data)
    num_layers = len(nn.shape)
    for i in range(1, num_layers):
        ax.scatter([i]*nn.shape[i], range(nn.shape[i]), color='blue', marker='o', s=100)
        ax.scatter([i-1]*nn.shape[i-1], range(nn.shape[i-1]), color='red', marker='o', s=100)
        for j in range(nn.shape[i]):
            for k in range(nn.shape[i-1]):
                ax.plot([i-1, i], [k, j], color='black', linewidth=0.5)

def visualize_backward_pass(ax, nn,training_data):
    # Visualize the backward pass calculations at each layer
    input_data = training_data[0][0]  # Take the first input data from the training data
    _, zs = nn.forward_pass(input_data)
    num_layers = len(nn.shape)
    for i in range(num_layers-1, 0, -1):
        ax.scatter([i]*nn.shape[i], range(nn.shape[i]), color='blue', marker='o', s=100)
        ax.scatter([i-1]*nn.shape[i-1], range(nn.shape[i-1]), color='red', marker='o', s=100)
        for j in range(nn.shape[i]):
            for k in range(nn.shape[i-1]):
                ax.plot([i-1, i], [k, j], color='black', linewidth=0.5)