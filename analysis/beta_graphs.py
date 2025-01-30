import numpy as np
import matplotlib.pyplot as plt

data = np.load('/mnt/ksf2/H1/user/u0100486/linux/doctorate/github/tracker_new/output/text_files/gamma_array_new.npy')
print(data)

# Extract unique values from column 1 (cm angle)
unique_angles = np.unique(data[:, 0])

# Create a single plot
plt.figure()

# Plot column 4 vs column 2 for each unique angle
for angle in unique_angles:
    # Filter rows where column 1 matches the current angle
    subset = data[data[:, 0] == angle]
    print(subset.shape)

    # Extract column 2 (x-axis) and column 4 (y-axis)
    x = subset[:, 1]  # Column 2
    y = subset[:, 2]  # Column 4

    # Plot with a label for the legend
    plt.plot(x, y, 'o-', label=f'CM Angle = {angle}')

# Add labels, title, legend, and grid
plt.xlabel('Column 2')
plt.ylabel('Column 4')
plt.title('Column 4 vs Column 2 for Different CM Angles')
plt.legend()
plt.grid()

# Show the plot
plt.show()