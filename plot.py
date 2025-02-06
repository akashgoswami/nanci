import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid points
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

# Calculate Z values (output of the function)
# Adding quadratic terms: y = 3x₁ - 2x₂ + 0.2x₁² + 0.1x₂²
Z = 3*X1 - 2*X2 + 0.2*X1**2 + 0.1*X2**2

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(X1, X2, Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('y')
ax.set_title('y = 3x₁ - 2x₂ + 0.2x₁² + 0.1x₂²')

# Add colorbar
plt.colorbar(surface)

plt.show()