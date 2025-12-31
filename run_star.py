"""
STaR Loop: generate_synth.py -> device_train.py -> repeat
Live graphing of results.csv
"""

import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RESULTS_PATH = "results.csv"


def update_graphs(frame):
    """Refresh both graphs from results.csv."""
    df = pd.read_csv(RESULTS_PATH)
    
    # Overall correctness
    ax1.clear()
    ax1.plot(df["iteration"], df["total_correct"] / df["total_examples"] * 100, "o-", color="#2ecc71", linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Overall Correctness (First + Rationalized)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Unconditioned (first attempt) correctness
    ax2.clear()
    ax2.plot(df["iteration"], df["first_correct"] / df["total_examples"] * 100, "o-", color="#3498db", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Unconditioned Answer Correctness (First Attempt)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)


def start_live_graphs():
    """Launch live updating matplotlib graphs in separate thread."""
    global fig, ax1, ax2
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("STaR Training Progress", fontsize=14)
    
    ani = FuncAnimation(fig, update_graphs, interval=5000, cache_frame_data=False)
    plt.tight_layout()
    plt.show(block=False)
    
    return ani


def run_generate(iteration):
    """Run generate_synth.py for given iteration."""
    print(f"\n[Iter {iteration}] Generating synthetic data...")
    subprocess.run([sys.executable, "generate_synth.py", "--iteration", str(iteration)])


def run_train():
    """Run device_train.py to train on latest synth data."""
    print(f"Training on synthetic data...")
    subprocess.run([sys.executable, "device_train.py"])


def get_current_iteration():
    """Read current iteration from results.csv."""
    df = pd.read_csv(RESULTS_PATH)
    return int(df["iteration"].max())


def main():
    ani = start_live_graphs()
    
    iteration = get_current_iteration() + 1
    
    while iteration <= 10:
        run_generate(iteration)
        run_train()
        iteration += 1
        
        plt.pause(0.1)  # Let graphs update
    
    print("\nSTaR training complete (10 iterations)")
    plt.ioff()
    plt.show()  # Keep graphs open at end


if __name__ == "__main__":
    main()
