import os
import re
import argparse
import matplotlib.pyplot as plt

def parse_training_log(training_log_path):
    """ Parses the training log file and extracts validation accuracy per epoch. """
    epochs = []
    val_acc1 = []

    with open(training_log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Match lines with epoch data (e.g., "0, 4.6973, 1.37, 4.5158, 3.01, 11.50")
        match = re.match(r"(\d+), [\d.]+, [\d.]+, [\d.]+, ([\d.]+), [\d.]+", line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            epochs.append(epoch)
            val_acc1.append(acc)

    return epochs, val_acc1

def plot_validation_accuracy(epochs, val_acc1, save_path):
    """ Plots validation accuracy per epoch and saves the figure. """
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, val_acc1, marker="o", linestyle="-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Over Epochs")
    plt.grid(True)
    plt.legend()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Parse training logs and plot validation accuracy over epochs.")
    parser.add_argument("--folder", type=str, help="Path to the folder containing training_log.txt")
    args = parser.parse_args()

    # Define the path to training_log.txt
    training_log_path = os.path.join(args.folder, "training_log.txt")

    # Check if the file exists
    if not os.path.exists(training_log_path):
        print(f"Error: {training_log_path} does not exist!")
        return

    # Parse the log file
    epochs, val_acc1 = parse_training_log(training_log_path)

    if not epochs or not val_acc1:
        print("No epoch validation accuracy data found!")
        return

    # Define save path for the plot
    save_path = os.path.join(args.folder, "val_accuracy_plot.png")

    # Generate and save the plot
    plot_validation_accuracy(epochs, val_acc1, save_path)

if __name__ == "__main__":
    main()