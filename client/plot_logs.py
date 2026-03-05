import re
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict

def parse_logs(log_file):
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return None

    # Regex to capture: [TAG] Iter | Reward | Acc
    # Example: [ANSWER-00]    1 |   1.10 |   60%
    metric_pattern = re.compile(r'^\[([\w-]+)\s*\]\s+(\d+)\s+\|\s+([-\d\.]+)\s+\|\s+([-\d\.]+)%?')
    
    data = defaultdict(lambda: {'iter': [], 'reward': [], 'acc': []})

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            match = metric_pattern.match(line)
            if match:
                tag = match.group(1)
                iteration = int(match.group(2))
                reward = float(match.group(3))
                # Handle accuracy with or without %
                acc_str = match.group(4)
                acc = float(acc_str)
                # If existing logs have 0-100 scale, normalize to 0-1 if specific logic requires, 
                # but typically we just plot what's there. 
                # The log shows "60%", "100%", so these are 0-100.
                
                data[tag]['iter'].append(iteration)
                data[tag]['reward'].append(reward)
                data[tag]['acc'].append(acc)

    return data

def plot_combined(data, output_file='combined_metrics.png'):
    if not data:
        print("No metrics found in log file.")
        return

    tags = sorted(data.keys())
    n_tags = len(tags)
    
    if n_tags == 0:
        print("No tags found.")
        return

    # Create subplots: 1 row per tag, 2 columns (Reward, Accuracy)
    # OR 1 row per tag, with dual axis? 
    # Let's do 1 row per tag, dual axis for compactness, or side-by-side?
    # User requested "separate graphs one row for each job".
    # Let's do 1 row per job, and in that row, maybe just plot Reward? Or Reward + Acc?
    # Let's plot Reward and Accuracy in parallel subplots or dual axis.
    # Let's try: N rows, 2 columns (Reward, Accuracy) covers everything clearly.
    
    fig, axes = plt.subplots(n_tags, 2, figsize=(12, 4 * n_tags), sharex=True)
    
    # Handle single row case (axes is 1D array)
    if n_tags == 1:
        axes = [axes]

    for i, tag in enumerate(tags):
        iters = data[tag]['iter']
        rewards = data[tag]['reward']
        accs = data[tag]['acc']
        
        # Reward Plot
        ax_reward = axes[i][0]
        ax_reward.plot(iters, rewards, 'b-o', label='Reward')
        ax_reward.set_title(f"Job: {tag} - Reward")
        ax_reward.set_ylabel("Reward")
        ax_reward.grid(True)
        
        # Accuracy Plot
        ax_acc = axes[i][1]
        ax_acc.plot(iters, accs, 'g-o', label='Accuracy')
        ax_acc.set_title(f"Job: {tag} - Accuracy")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0, 105) # Assume 0-100 scale
        ax_acc.grid(True)
        
        # Only set xlabel for bottom row
        if i == n_tags - 1:
            ax_reward.set_xlabel("Iteration")
            ax_acc.set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Combined metrics plot saved to {output_file}")
    
    # Close the figure to free memory after saving
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot parallel RLVR metrics from log file.')
    parser.add_argument('log_file', nargs='?', default='rlvr_parallel_results.log', 
                        help='Path to the rlvr_parallel_results.log file')
    parser.add_argument('--watch', action='store_true', help='Monitor log file and update plot continuously')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    
    args = parser.parse_args()

    if args.watch:
        import time
        print(f"Monitoring {args.log_file} every {args.interval} seconds...")
        try:
            while True:
                if not os.path.exists(args.log_file):
                    print(f"Waiting for log file: {args.log_file} (retrying in {args.interval}s)...", end='\r')
                else:
                    data = parse_logs(args.log_file)
                    if data:
                        plot_combined(data)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring.")
    else:
        data = parse_logs(args.log_file)
        plot_combined(data)
