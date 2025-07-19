import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define paths
results_dir = "/home/ubuntu/enhanced_simulation_results/"
figures_dir = "/home/ubuntu/figures/"
os.makedirs(figures_dir, exist_ok=True)

# Files to process
files = {
    "CFL-KBS": os.path.join(results_dir, "CFL_KBS_clusters4_topk2.json"),
    "FedAvg_HE": os.path.join(results_dir, "FedAvg_HE.json"),
    "FedMed_HE": os.path.join(results_dir, "FedMed_HE.json"),
    "FedProx_HE": os.path.join(results_dir, "FedProx_HE.json")
}

all_metrics_data = {}
final_metrics_list = []

# Load data
for name, path in files.items():
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            all_metrics_data[name] = data['metrics']
            # Add algorithm name to final metrics
            final_metrics = data['final_metrics']
            final_metrics['algorithm'] = name
            final_metrics_list.append(final_metrics)
            print(f"Loaded data for {name}")
    except FileNotFoundError:
        print(f"Error: File not found - {path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
    except KeyError as e:
        print(f"Error: Missing key {e} in {path}")

# Create DataFrame for final metrics
if not final_metrics_list:
    print("No final metrics data loaded. Exiting.")
    exit()

final_df = pd.DataFrame(final_metrics_list)
final_df.set_index('algorithm', inplace=True)

# --- Plot 1: Final Accuracy Comparison --- 
try:
    plt.figure(figsize=(10, 6))
    final_df['accuracy'].plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Comparaison de la Précision Finale (Données Synthétiques)')
    plt.ylabel('Précision')
    plt.xlabel('Algorithme')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_final_accuracy.png'))
    plt.close()
    print("Generated synthetic_final_accuracy.png")
except KeyError:
    print("Skipping final accuracy plot - 'accuracy' key missing.")
except Exception as e:
    print(f"Error generating final accuracy plot: {e}")

# --- Plot 2: Final F1 Score Comparison --- 
try:
    plt.figure(figsize=(10, 6))
    final_df['f1'].plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Comparaison du Score F1 Final (Données Synthétiques)')
    plt.ylabel('Score F1')
    plt.xlabel('Algorithme')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_final_f1_score.png'))
    plt.close()
    print("Generated synthetic_final_f1_score.png")
except KeyError:
    print("Skipping final F1 score plot - 'f1' key missing.")
except Exception as e:
    print(f"Error generating final F1 score plot: {e}")

# --- Plot 3: Final Loss Comparison --- 
try:
    plt.figure(figsize=(10, 6))
    final_df['loss'].plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Comparaison de la Perte Finale (Données Synthétiques)')
    plt.ylabel('Perte')
    plt.xlabel('Algorithme')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_final_loss.png'))
    plt.close()
    print("Generated synthetic_final_loss.png")
except KeyError:
    print("Skipping final loss plot - 'loss' key missing.")
except Exception as e:
    print(f"Error generating final loss plot: {e}")

# --- Plot 4: Communication Cost Comparison --- 
try:
    # Sum communication cost over rounds if it's a list, otherwise use final value
    comm_costs = {}
    for name, metrics in all_metrics_data.items():
        if 'communication_cost' in metrics and isinstance(metrics['communication_cost'], list):
            comm_costs[name] = sum(metrics['communication_cost'])
        elif 'communication_cost' in final_df.columns:
             comm_costs[name] = final_df.loc[name, 'communication_cost'] # Use final value if not list
        else:
             comm_costs[name] = 0 # Default if missing

    comm_df = pd.Series(comm_costs)
    plt.figure(figsize=(10, 6))
    comm_df.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Comparaison du Coût de Communication Total (Données Synthétiques)')
    plt.ylabel('Coût de Communication Total')
    plt.xlabel('Algorithme')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_communication_cost.png'))
    plt.close()
    print("Generated synthetic_communication_cost.png")
except KeyError:
    print("Skipping communication cost plot - 'communication_cost' key missing.")
except Exception as e:
    print(f"Error generating communication cost plot: {e}")

# --- Plot 5: Memory Usage Comparison --- 
try:
    # Use average memory usage over rounds if it's a list, otherwise use final value
    mem_usage = {}
    for name, metrics in all_metrics_data.items():
        if 'memory_usage' in metrics and isinstance(metrics['memory_usage'], list) and len(metrics['memory_usage']) > 0:
            mem_usage[name] = sum(metrics['memory_usage']) / len(metrics['memory_usage'])
        elif 'memory_usage' in final_df.columns:
             mem_usage[name] = final_df.loc[name, 'memory_usage'] # Use final value if not list
        else:
             mem_usage[name] = 0 # Default if missing

    mem_df = pd.Series(mem_usage)
    plt.figure(figsize=(10, 6))
    mem_df.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Comparaison de l\'Utilisation Mémoire Moyenne (Données Synthétiques)')
    plt.ylabel('Utilisation Mémoire Moyenne')
    plt.xlabel('Algorithme')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_memory_usage.png'))
    plt.close()
    print("Generated synthetic_memory_usage.png")
except KeyError:
    print("Skipping memory usage plot - 'memory_usage' key missing.")
except Exception as e:
    print(f"Error generating memory usage plot: {e}")

# --- Plot 6: Convergence Time Comparison --- 
try:
    # Use final convergence time
    plt.figure(figsize=(10, 6))
    final_df['convergence_time'].plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Comparaison du Temps de Convergence Total (Données Synthétiques)')
    plt.ylabel('Temps de Convergence Total (s)')
    plt.xlabel('Algorithme')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_convergence_time.png'))
    plt.close()
    print("Generated synthetic_convergence_time.png")
except KeyError:
    print("Skipping convergence time plot - 'convergence_time' key missing.")
except Exception as e:
    print(f"Error generating convergence time plot: {e}")

# --- Plot 7: Accuracy over Rounds --- 
try:
    plt.figure(figsize=(12, 7))
    for name, metrics in all_metrics_data.items():
        if 'accuracy' in metrics and isinstance(metrics['accuracy'], list):
            plt.plot(metrics['accuracy'], label=name)
    plt.title('Précision par Round (Données Synthétiques)')
    plt.xlabel('Round')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_accuracy_rounds.png'))
    plt.close()
    print("Generated synthetic_accuracy_rounds.png")
except KeyError:
    print("Skipping accuracy over rounds plot - 'accuracy' key missing or not a list.")
except Exception as e:
    print(f"Error generating accuracy over rounds plot: {e}")

# --- Plot 8: Loss over Rounds --- 
try:
    plt.figure(figsize=(12, 7))
    for name, metrics in all_metrics_data.items():
        if 'loss' in metrics and isinstance(metrics['loss'], list):
            plt.plot(metrics['loss'], label=name)
    plt.title('Perte par Round (Données Synthétiques)')
    plt.xlabel('Round')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'synthetic_loss_rounds.png'))
    plt.close()
    print("Generated synthetic_loss_rounds.png")
except KeyError:
    print("Skipping loss over rounds plot - 'loss' key missing or not a list.")
except Exception as e:
    print(f"Error generating loss over rounds plot: {e}")

print("\nVisualization script finished.")

