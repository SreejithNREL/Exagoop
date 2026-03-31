import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_results(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def group_by_dim(results):
    grouped = defaultdict(list)
    for entry in results:
        grouped[entry["dim"]].append(entry)
    return grouped

def plot_rms_by_dim(grouped):
    for dim, entries in grouped.items():
        # Sort by order for consistent plotting
        entries = sorted(entries, key=lambda x: x["order"])
        
        orders = [e["order"] for e in entries]
        rms_vals = [e["rms"] for e in entries]

        plt.figure(figsize=(6,4))
        plt.bar(orders, rms_vals, color="steelblue")
        plt.xlabel("Order")
        plt.ylabel("RMS Error")
        plt.title(f"RMS Error vs Order (dim = {dim})")
        plt.xticks(orders)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

def plot_combined(grouped):
    plt.figure(figsize=(7,5))

    for dim, entries in grouped.items():
        entries = sorted(entries, key=lambda x: x["order"])
        orders = [e["order"] for e in entries]
        rms_vals = [e["rms"] for e in entries]
        plt.plot(orders, rms_vals, marker="o", label=f"dim={dim}")

    plt.xlabel("Order")
    plt.ylabel("RMS Error")
    plt.title("RMS Error vs Order (All Dimensions)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    json_file = "sweep_results.json"  # <-- change if needed
    results = load_results(json_file)
    grouped = group_by_dim(results)

    plot_rms_by_dim(grouped)
    plot_combined(grouped)

