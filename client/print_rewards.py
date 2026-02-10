import re
rewards = []
with open("simple_rl_output.txt", "r") as f:
    for line in f:
        match = re.search(r"Rwd: ([+-]?\d+\.\d+)", line)
        if match:
            rewards.append(float(match.group(1)))

print("Total Epochs:", len(rewards))
print("First 20:", sum(rewards[:20])/20.0)
print("Last 20:", sum(rewards[-20:])/20.0)
