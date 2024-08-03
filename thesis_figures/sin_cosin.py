import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Create the plot
plt.figure(figsize=(6, 4))
plt.plot(x, y_sin, label='sin(x)', linewidth=5)
plt.plot(x, y_cos, label='cos(x)', linewidth=5)
# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Background transparency
plt.gca().patch.set_alpha(1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(fontsize=12)
plt.grid(False)  # Removing the grid
plt.savefig('thesis_figures/sin_cos.pdf', transparent=True)
plt.savefig('thesis_figures/sin_cos.png', transparent=True)

