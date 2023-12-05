import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

from numba import njit
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata

@njit
def run_simulation(audit_probability, audit_period, grid_size=100, social_pressure=25.0, simulation_steps=1000):
    # Set initial constants and parameters
    total_cells = grid_size * grid_size
    max_index = (grid_size + 2) * grid_size
    grid_size_incremented = grid_size + 1
    total_cells_incremented = grid_size * grid_size + grid_size
    twice_grid_size_incremented = 2 * grid_size + 1

    # Set initial states and memory to zero
    states = np.zeros((grid_size * (grid_size + 2),))
    transition_probabilities = np.zeros((9,), dtype=float)
    memory = np.zeros((grid_size * (grid_size + 2),))
    evader_fractions = []

    random.seed(1)  # Initialize random number generator. Use any number as the seed.

    # Initialization: Set everybody to honest and their memory to zero
    memory[:max_index] = 0
    states[:max_index] = 1

    # Spin-flip probabilities
    for interaction_energy in range(-4, 5, 2):
        energy_exp = np.exp(-interaction_energy * 2.0 / social_pressure)
        transition_probabilities[interaction_energy + 4] = (2.0 * energy_exp / (1.0 + energy_exp) - 1.0) * audit_probability

    # Simulate the dynamics of tax evasion over simulation_steps+1 time steps
    for sim_step in range(1, simulation_steps + 1):
        honest_count = 0
        for cell_index in range(grid_size_incremented, total_cells_incremented + 1):
            # First periodic border constraint
            if cell_index == twice_grid_size_incremented:
                states[total_cells_incremented:total_cells_incremented + grid_size] = states[grid_size:grid_size + grid_size]

            # Audited tax evaders must remain honest for audit_period periods
            if memory[cell_index - 1] > 0:
                memory[cell_index - 1] -= 1
                states[cell_index - 1] = 1
            else:
                interaction_energy = states[cell_index - 1] * (states[cell_index - 2] + states[cell_index] + states[cell_index - grid_size - 1] + states[cell_index + grid_size - 1])
                if random.uniform(-1, 1) < transition_probabilities[int(interaction_energy) + 4]:
                    states[cell_index - 1] = -states[cell_index - 1]

            # Counting the number of honest citizens
            if states[cell_index - 1] == 1:
                honest_count += 1

            # Each audited tax payer obtains a memory
            if random.uniform(-1, 1) < (2 * audit_probability - 1) and states[cell_index - 1] == -1:
                memory[cell_index - 1] = audit_period

        # Second periodic border constraint
        states[:grid_size] = states[total_cells:total_cells + grid_size]

        # The number of tax evaders
        evaders = total_cells - honest_count
        evader_fractions.append(evaders / float(total_cells))

    return evader_fractions

# Run the simulation for different audit probabilities and punishment lengths
audit_probabilities = np.linspace(0,1,21)
punishment_lengths = [10, 50]
results = {}
for audit_probability in audit_probabilities:
    for punishment_length in punishment_lengths:
        evader_fractions = run_simulation(audit_probability, punishment_length)
        results[(audit_probability, punishment_length)] = evader_fractions
        
fig = plt.figure(figsize=(10, 5))

for subplot, punishment_length in enumerate(punishment_lengths, 1):
    ax = fig.add_subplot(1, 2, subplot, projection='3d')

    # restructure data for surface plot
    X_data = []
    Y_data = []
    Z_data = []
    
    for audit_probability in audit_probabilities:
        evader_fractions = results[(audit_probability, punishment_length)]
        time_periods = range(len(evader_fractions))
        X_data.extend([audit_probability]*len(time_periods))
        Y_data.extend(time_periods)
        Z_data.extend(evader_fractions)
    
    # convert lists to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    Z_data = np.array(Z_data)

    # create x,y grid
    X_grid, Y_grid = np.meshgrid(np.unique(X_data), np.unique(Y_data))
    
    # interpolate Z values for grid
    Z_grid = griddata((X_data, Y_data), Z_data, (X_grid, Y_grid))

    # plot surface
    surf = ax.plot_surface(X_grid, Y_grid, Z_grid, rstride=5, cstride =5, linewidth=0, antialiased=False)
    ax.view_init(elev=20, azim=45)  # Modify these values to change the viewing angle
    ax.set_xlabel('Audit probability')
    ax.set_ylabel('Time period')
    ax.set_zlabel('Tax evasion portion')
    ax.set_title(f'Punishment length: {punishment_length}')

plt.tight_layout()
plt.show()

# 3D plots for each punishment length
fig = plt.figure(figsize=(10, 5))
for subplot, punishment_length in enumerate(punishment_lengths, 1):
    ax = fig.add_subplot(1, 2, subplot, projection='3d')
    for audit_probability in audit_probabilities:
        evader_fractions = results[(audit_probability, punishment_length)]
        time_periods = range(len(evader_fractions))
        ax.plot([audit_probability]*len(time_periods), time_periods, evader_fractions)
    ax.set_xlabel('Audit probability')
    ax.set_ylabel('Time period')
    ax.set_zlabel('Tax evasion portion')
    ax.set_title(f'Punishment length: {punishment_length}')

plt.tight_layout()
plt.show()

# 2D plots for specific time series
specific_audit_probabilities = [audit_probabilities[4], audit_probabilities[-3]]
fig, axes = plt.subplots(len(punishment_lengths), len(specific_audit_probabilities), figsize=(10, 10))
for j, punishment_length in enumerate(punishment_lengths):
    for i, audit_probability in enumerate(specific_audit_probabilities):
        ax = axes[i, j]
        evader_fractions = results[(audit_probability, punishment_length)]
        ax.plot(evader_fractions)
        ax.set_xlabel('Time period')
        ax.set_ylabel('Tax evasion portion')
        ax.set_title(f'Audit probability: {audit_probability}, Punishment length: {punishment_length}')

plt.tight_layout()
plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a subplot, and position the plots in a 1x2 grid
fig = make_subplots(rows=1, cols=2, subplot_titles=punishment_lengths, specs=[[{'type': 'surface'}, {'type': 'surface'}]])

for subplot, punishment_length in enumerate(punishment_lengths, 1):
    # restructure data for surface plot
    X_data = []
    Y_data = []
    Z_data = []
    
    for audit_probability in audit_probabilities:
        evader_fractions = results[(audit_probability, punishment_length)]
        time_periods = range(len(evader_fractions))
        X_data.extend([audit_probability]*len(time_periods))
        Y_data.extend(time_periods)
        Z_data.extend(evader_fractions)
    
    # convert lists to numpy arrays
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    Z_data = np.array(Z_data)

    # create x,y grid
    X_grid, Y_grid = np.meshgrid(np.unique(X_data), np.unique(Y_data))
    
    # interpolate Z values for grid
    Z_grid = griddata((X_data, Y_data), Z_data, (X_grid, Y_grid))

    # Create the surface trace
    trace = go.Surface(x=X_grid, y=Y_grid, z=Z_grid, colorscale='Greys', showscale=False)
    
    # Add surface trace data to figure
    fig.add_trace(trace, row=1, col=subplot)

    # update layout
    fig.update_layout(
        title=f"Tax Evasion Simulation - Punishment length: {punishment_length}",
        scene = dict(
            xaxis_title='Audit probability',
            yaxis_title='Time period',
            zaxis_title='Tax evasion portion'
        ),
        autosize=False,
        width=1000, 
        height=500,
        margin=dict(l=50, r=50, b=65, t=90)
    )

fig.show()
