import os
import matplotlib.pyplot as plt

def get_data(exp_id):
    path = f"mlruns/{exp_id}"
    runs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d != 'models']
    if not runs: return None
    
    # Get accuracy from the first run found
    acc_path = os.path.join(path, runs[0], "metrics", "accuracy")
    if os.path.exists(acc_path):
        with open(acc_path, "r") as f:
            return [float(line.split()[1]) for line in f.readlines()]
    return None

vgg = get_data("1")
mob = get_data("2")

plt.figure(figsize=(10, 6))
if vgg: plt.plot(vgg, label="VGG16 (Experiment 1)", linewidth=2)
if mob: plt.plot(mob, label="MobileNet (Experiment 2)", linewidth=2)

plt.title("Model Training Progress - Politician Classifier", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Epochs", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("final_accuracy_report.png")
print("✅ SUCCESS: Graph saved as 'final_accuracy_report.png'. Use this for your report!")