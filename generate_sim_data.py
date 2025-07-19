import pandas as pd
import numpy as np
import json
import os
from sklearn.datasets import make_classification

# Simulation Parameters
NUM_CLIENTS = 30 # Increased to 30
NUM_CLUSTERS = 4
# Added PE (Polymorphic Encryption) and SMPC (Secure Multi-Party Computation)
ENCRYPTION_SCHEMES = ["AES", "RSA", "ECC", "HE", "PE", "SMPC"]
COMPUTATIONAL_POWER_LEVELS = ["low", "medium", "high"]
DATA_SENSITIVITY_LEVELS = ["low", "medium", "high"]
NUM_FEATURES = 50 # Example number of features for synthetic data
NUM_CLASSES = 6 # Example number of classes (like HAR)
SAMPLES_PER_CLIENT = 200 # Average samples per client
DIRICHLET_ALPHA = 0.5 # Controls non-IID level (lower alpha = more non-IID)
DATA_DIR = "/home/ubuntu/synthetic_data"

# Seed for reproducibility
np.random.seed(42)

# --- Create Data Directory ---
os.makedirs(DATA_DIR, exist_ok=True)

# --- Generate Client Data ---
client_data = []
for i in range(NUM_CLIENTS):
    client_id = f"client_{i+1}"
    comp_power = np.random.choice(COMPUTATIONAL_POWER_LEVELS, p=[0.4, 0.4, 0.2])
    data_sensitivity = np.random.choice(DATA_SENSITIVITY_LEVELS, p=[0.3, 0.5, 0.2])
    base_latency = np.random.uniform(50, 500)
    base_bandwidth = np.random.uniform(1, 100)

    client_data.append({
        "client_id": client_id,
        "computational_power": comp_power,
        "data_sensitivity": data_sensitivity,
        "base_latency_ms": round(base_latency, 2),
        "base_bandwidth_mbps": round(base_bandwidth, 2)
    })

clients_df = pd.DataFrame(client_data)

# --- Generate Synthetic Training Data (Non-IID) ---
print(f"Generating non-IID synthetic data for {NUM_CLIENTS} clients...")
# Generate overall dataset statistics (can be adjusted)
X_global, y_global = make_classification(
    n_samples=NUM_CLIENTS * SAMPLES_PER_CLIENT,
    n_features=NUM_FEATURES,
    n_informative=int(NUM_FEATURES * 0.6),
    n_redundant=int(NUM_FEATURES * 0.1),
    n_classes=NUM_CLASSES,
    n_clusters_per_class=2,
    random_state=42
)

# Distribute data non-IID using Dirichlet distribution
client_data_indices = {i: [] for i in range(NUM_CLIENTS)}
label_distribution = np.random.dirichlet([DIRICHLET_ALPHA] * NUM_CLIENTS, NUM_CLASSES)

# Assign data points to clients based on label distribution
for k in range(NUM_CLASSES):
    idx_k = np.where(y_global == k)[0]
    np.random.shuffle(idx_k)
    # Proportions of class k for each client
    proportions = label_distribution[k]
    # Cumulative proportions
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    # Split indices based on proportions
    client_splits = np.split(idx_k, proportions)
    for i in range(NUM_CLIENTS):
        client_data_indices[i].extend(client_splits[i])

# Save data for each client
for i in range(NUM_CLIENTS):
    client_id = f"client_{i+1}"
    indices = client_data_indices[i]
    X_client = X_global[indices]
    y_client = y_global[indices]

    # Save as CSV (or other format like .npy)
    client_df = pd.DataFrame(X_client, columns=[f"feature_{j}" for j in range(NUM_FEATURES)])
    client_df["label"] = y_client
    client_df.to_csv(os.path.join(DATA_DIR, f"{client_id}_data.csv"), index=False)

print(f"Synthetic data saved in {DATA_DIR}")

# --- Generate Encryption Performance Data (Simulated) ---
# Added PE and SMPC with estimated relative costs
perf_data = {
    "AES": {"compute_factor": 1, "security_score": 6, "size_factor": 1, "memory_factor": 1},
    "RSA": {"compute_factor": 5, "security_score": 7, "size_factor": 3, "memory_factor": 2},
    "ECC": {"compute_factor": 8, "security_score": 8, "size_factor": 2, "memory_factor": 3},
    "HE": {"compute_factor": 20, "security_score": 9, "size_factor": 10, "memory_factor": 5},
    "PE": {"compute_factor": 15, "security_score": 8.5, "size_factor": 5, "memory_factor": 4}, # Polymorphic Encryption (estimated)
    "SMPC": {"compute_factor": 10, "security_score": 7.5, "size_factor": 15, "memory_factor": 3} # Secure Multi-Party Computation (high comms cost)
}

power_modifier = {"low": 1.5, "medium": 1.0, "high": 0.7}
sensitivity_modifier = {"low": 0.8, "medium": 1.0, "high": 1.2}

encryption_performance = []
for _, client in clients_df.iterrows():
    client_id = client["client_id"]
    comp_power = client["computational_power"]
    data_sens = client["data_sensitivity"]

    for scheme in ENCRYPTION_SCHEMES:
        base = perf_data[scheme]
        compute_time = base["compute_factor"] * power_modifier[comp_power] * np.random.uniform(0.9, 1.1)
        security = base["security_score"] * sensitivity_modifier[data_sens] * np.random.uniform(0.95, 1.05)
        comm_size = base["size_factor"] * np.random.uniform(0.8, 1.2)
        memory_usage = base["memory_factor"] * power_modifier[comp_power] * np.random.uniform(0.9, 1.1)

        encryption_performance.append({
            "client_id": client_id,
            "scheme": scheme,
            "compute_time_factor": round(compute_time, 2),
            "security_score": round(security, 2),
            "comm_size_factor": round(comm_size, 2),
            "memory_factor": round(memory_usage, 2)
        })

encryption_df = pd.DataFrame(encryption_performance)

# --- Save Data --- #
clients_df.to_csv("/home/ubuntu/sim_clients.csv", index=False)
encryption_df.to_csv("/home/ubuntu/sim_encryption_performance.csv", index=False)

# Updated metrics list
sim_params = {
    "num_clients": NUM_CLIENTS,
    "num_clusters": NUM_CLUSTERS,
    "encryption_schemes": ENCRYPTION_SCHEMES,
    "computational_power_levels": COMPUTATIONAL_POWER_LEVELS,
    "data_sensitivity_levels": DATA_SENSITIVITY_LEVELS,
    "metrics_to_track": [
        "accuracy",
        "communication_cost",
        "memory_used",
        "entropy",
        "f1_score",
        "recall",
        "precision",
        "convergence_time",
        "model_size",
        "global_loss",
        "encryption_switch_rate",
        "strategy_stability"
    ],
    "data_dir": DATA_DIR,
    "num_features": NUM_FEATURES,
    "num_classes": NUM_CLASSES
}
with open("/home/ubuntu/sim_params.json", "w") as f:
    json.dump(sim_params, f, indent=4)

print("Simulated data generated and saved to sim_clients.csv, sim_encryption_performance.csv, sim_params.json, and synthetic data files in /home/ubuntu/synthetic_data/")



# --- Generate Global Test Data ---
print("Generating global test data...")
NUM_TEST_SAMPLES = int(NUM_CLIENTS * SAMPLES_PER_CLIENT * 0.2) # e.g., 20% of total training samples
X_test, y_test = make_classification(
    n_samples=NUM_TEST_SAMPLES,
    n_features=NUM_FEATURES,
    n_informative=int(NUM_FEATURES * 0.6),
    n_redundant=int(NUM_FEATURES * 0.1),
    n_classes=NUM_CLASSES,
    n_clusters_per_class=2,
    random_state=123 # Use a different seed for test data
)
test_df = pd.DataFrame(X_test, columns=[f"feature_{j}" for j in range(NUM_FEATURES)])
test_df["label"] = y_test
test_df.to_csv(os.path.join(DATA_DIR, "global_test_data.csv"), index=False)
print(f"Global test data saved to {os.path.join(DATA_DIR, 'global_test_data.csv')}")

