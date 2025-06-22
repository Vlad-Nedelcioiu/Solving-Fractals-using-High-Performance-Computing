import re
import pandas as pd
import matplotlib.pyplot as plt

# Define log files
log_files = {
    "static": "logs\log_static.txt",
    "dynamic": "logs\log_dynamic.txt",
    "guided": "logs\log_guided.txt"
}

# Extract chunk data from logs
pattern = re.compile(r"Rows\s+(\d+)-(\d+):\s+([\d.]+) seconds")
data = []

for schedule, path in log_files.items():
    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                start, end, time = match.groups()
                data.append({
                    "schedule": schedule,
                    "chunk_id": int(start) // 200,         # this will be Y axis value
                    "label": f"{start}-{end}",             # what you want to show on Y axis
                    "time": float(time)
                })

df = pd.DataFrame(data)

# Get sorted unique chunks for labeling Y-axis
chunk_ticks = sorted(df["chunk_id"].unique())
chunk_labels = df.drop_duplicates("chunk_id").sort_values("chunk_id")["label"].tolist()

# Plot: time vs chunk_id, labeled Y axis
plt.figure(figsize=(12, 6))
for sched in df["schedule"].unique():
    chunk = df[df["schedule"] == sched]
    plt.plot(chunk["time"], chunk["chunk_id"], marker='o', label=sched)

# Custom Y ticks
plt.yticks(chunk_ticks, chunk_labels)

# Labels
plt.title("Row Progression Over Execution Time (Y = Chunk Labels)")
plt.xlabel("Execution Time (seconds)")
plt.ylabel("Row Chunk")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("execution_plot_custom_ylabels.png")
plt.show()
