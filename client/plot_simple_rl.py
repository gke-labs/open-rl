import re
import matplotlib.pyplot as plt

rewards_history = []
losses_history = []

with open("simple_rl_output.txt", "r") as f:
    for line in f:
        match = re.search(r"Epoch \d+ \| Tgt: .* \| Rwd: ([+-]?\d+\.\d+) \| Adv: .* \| Loss: ([+-]?\d+\.\d+)", line)
        if match:
            rewards_history.append(float(match.group(1)))
            losses_history.append(float(match.group(2)))

epochs = len(rewards_history)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)

window_size = max(1, epochs // 10)
smoothed_rewards = []
for i in range(epochs):
    start = max(0, i - window_size + 1)
    window = rewards_history[start:i+1]
    smoothed_rewards.append(sum(window) / len(window))
    
plt.plot(range(epochs), smoothed_rewards, linestyle='-', color='g', label='Mean Reward (SMA)')
plt.plot(range(epochs), rewards_history, marker='o', linestyle='', color='g', alpha=0.2, label='Step Reward')
plt.title('Simple Echo RL Reward')
plt.xlabel('Epoch')
plt.ylabel('Reward (-1.0 to 1.0)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses_history, marker='+', linestyle='-', color='r', alpha=0.6)
plt.title('Policy Gradient Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.savefig('simple_rl_rewards.png')
print("Successfully saved 'simple_rl_rewards.png'")
