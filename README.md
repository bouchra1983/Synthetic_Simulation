
This report presents the results of a series of simulations aimed at evaluating the performance of the CFL-KBS framework (Clustered Federated Learning with Knowledge-Based Security) in comparison with other standard federated learning (FL) algorithms: FedAvg, FedMed, and FedProx. Unlike previous simulations based on the HAR dataset, this evaluation uses a synthetically generated dataset specifically designed for this test.

1- Synthetic Data Generation
To simulate a heterogeneous federated learning (FL) environment, a synthetic dataset was generated using the script "generate_sim_data.py". The key characteristics of this dataset are as follows:

- Number of Clients: 30

- Number of Features: 50

- Number of Classes: 6 (labels ranging from 0 to 5)

- Data Size per Client: Variable (between 50 and 150 samples per client), to reflect heterogeneity in data volume.

- Class Distribution: Non-IID (Non-Independent and Identically Distributed). Each client holds a potentially unbalanced subset of the six global classes, simulating real-world data heterogeneity.

- Global Test Set: A separate global test dataset (global_test_data.csv) was also generated, containing 1,200 samples for consistent evaluation of the final global models.

The data files were saved in the directory "/Synthetic_Simulation/synthetic_data.rar/" in CSV format ("client_N_data.csv" for each client's training set and "global_test_data.csv" for global evaluation).

2- Simulation Configuration
The simulations were conducted using the "enhanced_cfl_simulation.py" script, which was modified to operate on the synthetic dataset. The key simulation parameters were as follows:

- Number of Rounds: 50

- Clients per Round: 20 (randomly selected in each round)

- Learning Model: Logistic Regression (adapted for 50 features and 6 classes)

- Compared Algorithms:

- CFL-KBS: Configured with 4 clusters and Top-2 encryption scheme selection for adaptive security.

- FedAvg: Baseline algorithm using fixed Homomorphic Encryption (HE).

- FedMed: A variant of FedAvg using geometric median aggregation, with HE.

- FedProx: A variant of FedAvg incorporating a proximal regularization term, also using HE.


