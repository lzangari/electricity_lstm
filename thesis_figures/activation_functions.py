import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, x * alpha)

def relu(x):
    return np.maximum(0, x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def step_function(x, theta=0):
    return np.where(x >= theta, 1, 0)

# Derivatives of activation functions
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def step_function_derivative(x, theta=0):
    return np.zeros_like(x)  # The step function is not differentiable, but we'll use zero for visualization

# Function to adjust axes and add arrows
def adjust_axes(ax):
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Move bottom spine to zero
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#D0D3D6')
    
    # Move left spine to zero
    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('#D0D3D6')
    
    # Add arrows
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, color='#D0D3D6', markersize=10 )
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, color='#D0D3D6', markersize=10 )
    
    # Make the axis thicker
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    # Limit view
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.2, 1.2)

# Generalize colors
line_color = "#423B64"
derivative_color = "#8785BA"

# Plotting function
def plot_activation_functions(order):
    x = np.linspace(-5, 5, 100)

    funcs = {
        'Sigmoid': (sigmoid, sigmoid_derivative),
        'Tanh': (tanh, tanh_derivative),
        'Leaky ReLU': (leaky_relu, leaky_relu_derivative),
        'ReLU': (relu, relu_derivative),
        'ELU': (elu, elu_derivative),
        'Step Function': (step_function, step_function_derivative)
    }

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()
    for i, name in enumerate(order):
        func, deriv = funcs[name]
        # Main plot for the activation function
        line1, = axes[i].plot(x, func(x), label='$f(x)$', linewidth=4, color=line_color)
        adjust_axes(axes[i])
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_xaxis().set_ticklabels([])
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_yaxis().set_ticklabels([])

        line2, = axes[i].plot(x, deriv(x), label='$f\'(x)$', linestyle='dotted', linewidth=4, color=derivative_color)

        if i == len(order) - 1:
        # Create a legend for the last plot with custom handles
            from matplotlib.legend import Legend
            leg = Legend(axes[i], [line1, line2], ['$f(x)$', '$f\'(x)$'], loc='lower right', frameon=False, fontsize=12)
            axes[i].add_artist(leg)
        # Transparent background of axes
        axes[i].patch.set_alpha(0)

    plt.tight_layout()
    # set transparent background
    fig.patch.set_alpha(0)
    plt.savefig('activation_functions_ordered.svg')
    plt.savefig('activation_functions_ordered.png')

# Define the order of functions to be plotted
order = ['Step Function', 'Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', ]
plot_activation_functions(order)
