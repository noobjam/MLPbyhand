import streamlit as st
import numpy as np
from neural_network import NeuralNetwork  # Import your neural network implementation
# Larger training dataset
training_data = []
for _ in range(1000):  # Generate 1000 random training examples
    input_data = np.random.rand(10)  # Example input data with 10 features
    target_data = np.random.rand(5)  # Example target data with 5 outputs
    training_data.append((input_data, target_data))
# Function to display initial state as text boxes
def display_initial_state(nn):
    st.subheader("Initial Data:")
    st.text("Shape of Input Data: " + str(training_data[0][0].shape))  # Assuming training_data is defined
    st.subheader("Initial Model Configuration:")
    st.text("Model Shape: " + str(nn.shape))
    st.subheader("Initial Weights:")
    for i, w in enumerate(nn.weights):
        st.text(f"Layer {i+1} Weights: {w}")
    st.subheader("Initial Biases:")
    for i, b in enumerate(nn.biases):
        st.text(f"Layer {i+1} Biases: {b}")
        
    st.subheader("the activations: ")
    st.text(str(nn.activations))

# Create a Streamlit app
st.title('Neural Network Initial State')

# Initialize neural network
shape = (10, 4, 5)  
nn = NeuralNetwork(shape)  # Initialize your neural network with the specified architecture

# Display initial state before pressing the button
display_initial_state(nn)
