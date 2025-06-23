import numpy as np
import matplotlib.pyplot as plt
import os

# Define the file path
file_path = 'Data/exampleLoss360Hops.npy'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

try:
    # Load the loss data from the .npy file
    loss_data = np.load(file_path)

    # The index will represent epochs.
    # If loss_data is 1D, np.arange(len(loss_data)) gives the epoch numbers.
    epochs = np.arange(len(loss_data))

    # Create the plot
    plt.figure(figsize=(10, 6)) # Adjust figure size as needed
    plt.plot(epochs, loss_data, linestyle='-', linewidth=6, color='blue') # Plot with circles and a line

    plt.rcParams.update({
        'font.size': 16,          # Default font size
        'axes.titlesize': 14,     # Title size
        'axes.labelsize': 1200,     # Axis label size
        'xtick.labelsize': 20,    # X-tick label size
        'ytick.labelsize': 20,    # Y-tick label size
        'legend.fontsize': 17     # Legend font size
    })

    # Add labels and title
    #plt.title('Training Loss over Epochs (360 Hops)')
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Loss', fontsize = 20)
    plt.grid(True) # Add a grid for readability

    # Show the plot
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred while loading or plotting the data: {e}")