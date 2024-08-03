import matplotlib.pyplot as plt
import json

def hours_to_days(hours):
    return [hour / 24 for hour in hours]

# File paths
files = {
    "Naive LSTM": ["model_info/lstm_naive_hour_losses.json", r"$M_{Naive}$", "#8785BA", "*", "solid", 7],
    "Stacked LSTM": ["model_info/lstm_stacked_hour_losses.json", r"$M_{Stacked}$", "#C195C4", "d", "dashdot", 1],
    "Sequence 2 Sequence Additive LSTM without regularization": ["model_info/lstm_seq2seq_additive_no_regularization_hour_losses.json", r"$M_{Seq2Seq}$", "#95AAD3", "^", "dashed", 1],
    "Sequence 2 Sequence Additive LSTM": ["model_info/lstm_seq2seq_additive_hour_losses.json", r"$M_{RegSeq2Seq}$", "#06948E", "v", "dotted", 2],
}

# Data dictionary to hold model data
data = {}

# Load data from JSON files
for model, (file_path, label, color, marker, linestyle, minimum) in files.items():
    with open(file_path, 'r') as file:
        data[model] = {"content": json.load(file), "label": label, "color": color, "marker": marker, "linestyle": linestyle, "minimum": minimum}

# Plotting
plt.figure(figsize=(6, 4))

for model, details in data.items():
    days = hours_to_days(details["content"]["seq_lengths"])
    losses = details["content"]["losses"]
    plt.plot(days, losses, label=details["label"], color=details["color"], linestyle=details["linestyle"], markersize=0, linewidth=1, alpha=.7)  # Set markersize to 0 for the line

    # Find the minimum loss to highlight
    min_loss_index = losses.index(min(losses))

    # Plot each point, highlight the minimum with larger marker size
    for i, (day, loss) in enumerate(zip(days, losses)):
        if i == min_loss_index:
            plt.plot(day, loss, marker=details["marker"], markersize=8, color=details["color"])  # Larger marker for minimum
        else:
            plt.plot(day, loss, marker=details["marker"], markersize=5, color=details["color"], alpha=.7)  # Normal marker size


# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel('Sequence Length (days)')
plt.ylabel('Loss')
plt.xticks(days)

# Retrieve and bold first, second, and last x-tick labels
xticks = plt.gca().get_xticklabels()
if len(xticks) >= 2:  # Check to ensure there are enough x-ticks to modify
    xticks[0].set_fontweight('bold')
    xticks[1].set_fontweight('bold')
    # for all the other x-ticks, set the font weight to normal
    for xtick in xticks[2:-1]:
        xtick.set_fontweight('normal')
    xticks[-1].set_fontweight('bold')

# Plot the legend center upper
plt.legend(loc='upper center', ncol=2)
# Save the plot
plt.savefig("thesis_figures/loss_vs_seq.png", dpi=300, bbox_inches='tight')
plt.savefig("thesis_figures/loss_vs_seq.pdf", dpi=300, bbox_inches='tight')
