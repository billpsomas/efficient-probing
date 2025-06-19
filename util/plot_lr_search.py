import os
import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_training_logs(parent_dir):
    """ Parses training logs from all trials and collects base learning rates with max accuracies. """
    blr_to_max_acc = defaultdict(list)

    # Iterate through all trial directories
    for trial in os.listdir(parent_dir):
        trial_path = os.path.join(parent_dir, trial)
        training_log_path = os.path.join(trial_path, "training_log.txt")

        # Check if the training_log.txt exists
        if os.path.exists(training_log_path):
            with open(training_log_path, "r") as f:
                lines = f.readlines()

            # Extract base learning rate and max accuracy
            blr = None
            max_acc = None

            for line in lines:
                if "Base learning rate:" in line:
                    blr = float(line.split(":")[1].strip())
                if "Max Accuracy:" in line:
                    max_acc = float(re.findall(r"[\d.]+", line)[0])  # Extract the accuracy value

            # Store the max accuracy corresponding to each base learning rate
            if blr is not None and max_acc is not None:
                blr_to_max_acc[blr].append(max_acc)

    return blr_to_max_acc

def plot_results(blr_to_max_acc, save_path):
    """ Plots max accuracy vs base learning rate and saves the figure. """
    # Compute the average max accuracy for each base learning rate
    blr_avg_max_acc = {blr: sum(acc_list) / len(acc_list) for blr, acc_list in blr_to_max_acc.items()}

    # Sort by base learning rate
    blr_sorted = sorted(blr_avg_max_acc.keys())
    acc_sorted = [blr_avg_max_acc[blr] for blr in blr_sorted]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(blr_sorted, acc_sorted, marker="o", linestyle="-")
    plt.xlabel("Base Learning Rate")
    plt.ylabel("Max Accuracy (%)")
    plt.title("Max Accuracy vs Base Learning Rate")
    plt.grid(True)
    plt.xscale("log")  # Log scale for better visualization

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Parse training logs and plot max accuracy vs base learning rate.")
    parser.add_argument("--parent_folder", type=str, help="Path to the parent directory containing trial folders.")
    args = parser.parse_args()

    # Parse training logs
    blr_to_max_acc = parse_training_logs(args.parent_folder)

    # Define save path for the figure
    save_path = os.path.join(args.parent_folder, "max_accuracy_vs_blr.png")

    # Plot results and save the figure
    plot_results(blr_to_max_acc, save_path)

if __name__ == "__main__":
    main()