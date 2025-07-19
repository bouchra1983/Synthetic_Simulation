import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import random
from collections import defaultdict
import math

# Définition des constantes
NUM_ROUNDS = 50
NUM_CLIENTS = 30  # Nombre total de clients disponibles
CLIENTS_PER_ROUND = 20  # Nombre de clients sélectionnés par round
SEED = 42
HAR_CLASSES = [0, 1, 2, 3, 4, 5] # 6 classes in HAR dataset (0-indexed)
NUM_FEATURES = 50 # Number of features in synthetic data

# Configuration des schémas de chiffrement
ENCRYPTION_SCHEMES = {
    # ... (Encryption schemes remain the same)
    'AES': {
        'security_level': 0.85,
        'computation_cost': 0.15,
        'communication_cost': 0.20,
        'memory_usage': 0.10,
        'entropy': 0.80
    },
    'RSA': {
        'security_level': 0.90,
        'computation_cost': 0.40,
        'communication_cost': 0.35,
        'memory_usage': 0.30,
        'entropy': 0.85
    },
    'ECC': {
        'security_level': 0.88,
        'computation_cost': 0.25,
        'communication_cost': 0.15,
        'memory_usage': 0.20,
        'entropy': 0.82
    },
    'HE': {
        'security_level': 0.95,
        'computation_cost': 0.80,
        'communication_cost': 0.70,
        'memory_usage': 0.75,
        'entropy': 0.90
    },
    'SMPC': {
        'security_level': 0.92,
        'computation_cost': 0.60,
        'communication_cost': 0.50,
        'memory_usage': 0.45,
        'entropy': 0.88
    },
    'DP': {
        'security_level': 0.82,
        'computation_cost': 0.30,
        'communication_cost': 0.25,
        'memory_usage': 0.15,
        'entropy': 0.75
    },
    'PE': {  # Polymorphic Encryption
        'security_level': 0.87,
        'computation_cost': 0.45,
        'communication_cost': 0.40,
        'memory_usage': 0.35,
        'entropy': 0.83
    }
}

# Chemins des données
DATA_DIR = "/home/ubuntu/synthetic_data/"
OUTPUT_DIR = "/home/ubuntu/enhanced_simulation_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper function for model initialization ---
def initialize_logistic_regression(num_features, classes):
    """Initializes and fits a Logistic Regression model on dummy data covering all classes."""
    model = LogisticRegression(max_iter=100, random_state=SEED, solver='saga', penalty='l2', C=1.0, tol=0.01, multi_class='ovr') # Use 'ovr' for multi-class
    # Create dummy data with at least one sample per class
    num_classes = len(classes)
    X_dummy = np.random.rand(num_classes * 2, num_features) # Ensure enough samples
    y_dummy = np.array(list(classes) * 2) # Ensure all classes are present
    try:
        model.fit(X_dummy, y_dummy)
    except Exception as e:
        print(f"Error during initial model fit: {e}")
        # Fallback if fit fails (should not happen with multi_class='ovr')
        pass 
    return model

# Configuration des algorithmes
class FedAvg:
    def __init__(self, num_clients, clients_per_round, num_rounds, encryption='HE'):
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.num_rounds = num_rounds
        self.encryption = encryption
        self.client_models = {}
        self.global_model = None
        self.metrics = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'loss': [],
            'communication_cost': [], 'memory_usage': [], 'entropy': [],
            'convergence_time': [], 'model_size': []
        }

    def initialize_clients(self):
        """Initialise les modèles pour chaque client"""
        for client_id in range(1, self.num_clients + 1):
            # Initialize model but don't fit yet
            self.client_models[client_id] = LogisticRegression(max_iter=100, random_state=SEED, solver='saga', penalty='l2', C=1.0, tol=0.01, multi_class='ovr')

    def select_clients(self, round_num):
        """Sélectionne un sous-ensemble de clients pour participer au round actuel"""
        return random.sample(range(1, self.num_clients + 1), self.clients_per_round)

    def train_client(self, client_id, X_train, y_train):
        """Entraîne le modèle d'un client sur ses données locales"""
        start_time = time.time()
        # Ensure the client model is initialized with correct classes before fitting
        if not hasattr(self.client_models[client_id], 'classes_'):
             self.client_models[client_id].classes_ = np.array(HAR_CLASSES)
             
        # Check if training data has at least 2 classes
        unique_classes_in_data = np.unique(y_train)
        if len(unique_classes_in_data) < 2:
            print(f"Warning: Client {client_id} training data has only {len(unique_classes_in_data)} class(es). Skipping fit.")
            # Optionally, return 0 time or handle differently
            return time.time() - start_time # Return time even if skipped
            
        try:
            self.client_models[client_id].fit(X_train, y_train)
        except ValueError as e:
             print(f"Error fitting client {client_id} model: {e}. Data shape: {X_train.shape}, Labels: {np.unique(y_train)}")
             # Handle error, maybe skip this client's update
             return time.time() - start_time # Return time even if error
             
        return time.time() - start_time

    def aggregate_models(self, selected_clients, client_weights):
        """Agrège les modèles des clients sélectionnés pour mettre à jour le modèle global"""
        if self.global_model is None:
            # Initialize the global model using the helper function
            self.global_model = initialize_logistic_regression(NUM_FEATURES, HAR_CLASSES)

        # Agréger les coefficients et l'intercept
        coef_sum = np.zeros_like(self.global_model.coef_)
        intercept_sum = np.zeros_like(self.global_model.intercept_)

        total_weight = sum(client_weights.values())
        if total_weight == 0:
            print("Warning: Total weight is zero during aggregation.")
            return

        num_aggregated = 0
        for client_id in selected_clients:
            # Ensure client model is fitted and has the correct shape before accessing coef_ and intercept_
            if hasattr(self.client_models[client_id], 'coef_') and self.client_models[client_id].coef_.shape == self.global_model.coef_.shape:
                weight = client_weights[client_id] / total_weight
                coef_sum += weight * self.client_models[client_id].coef_
                intercept_sum += weight * self.client_models[client_id].intercept_
                num_aggregated += 1
            else:
                print(f"Warning: Client {client_id} model not fitted or shape mismatch, skipping aggregation.")
                if hasattr(self.client_models[client_id], 'coef_'):
                     print(f"Client shape: {self.client_models[client_id].coef_.shape}, Global shape: {self.global_model.coef_.shape}")

        if num_aggregated > 0:
            self.global_model.coef_ = coef_sum
            self.global_model.intercept_ = intercept_sum
        else:
             print("Warning: No models were aggregated in this round.")

    def evaluate_model(self, X_test, y_test):
        """Évalue le modèle global sur les données de test"""
        if self.global_model is None or not hasattr(self.global_model, 'predict_proba'):
             print("Warning: Global model not initialized or fitted for evaluation.")
             return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': np.inf}

        try:
            y_pred = self.global_model.predict(X_test)
            y_proba = self.global_model.predict_proba(X_test)
        except Exception as e:
             print(f"Error during prediction: {e}")
             return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': np.inf}

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        try:
            # Ensure labels cover all possible classes in y_test and model's classes
            all_possible_labels = np.union1d(HAR_CLASSES, np.unique(y_test))
            loss = log_loss(y_test, y_proba, labels=all_possible_labels)
        except ValueError as e:
            print(f"Warning: Could not calculate log_loss: {e}. y_test unique: {np.unique(y_test)}, model classes: {self.global_model.classes_}")
            loss = np.inf
        except Exception as e:
             print(f"Unexpected error calculating log_loss: {e}")
             loss = np.inf

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'loss': loss}

    # --- Other methods (calculate_communication_cost, etc.) remain largely the same --- 
    def calculate_communication_cost(self, selected_clients):
        base_cost = len(selected_clients) * 0.1
        encryption_factor = ENCRYPTION_SCHEMES[self.encryption]['communication_cost']
        return base_cost * encryption_factor

    def calculate_memory_usage(self, selected_clients):
        base_usage = len(selected_clients) * 0.1
        encryption_factor = ENCRYPTION_SCHEMES[self.encryption]['memory_usage']
        return base_usage * encryption_factor

    def calculate_entropy(self):
        return ENCRYPTION_SCHEMES[self.encryption]['entropy']

    def get_model_size(self):
        if self.global_model is not None and hasattr(self.global_model, 'coef_'):
            return self.global_model.coef_.size + self.global_model.intercept_.size
        return 0
        
    def run_simulation(self):
        print(f"Démarrage de la simulation {self.__class__.__name__} avec chiffrement {self.encryption}")
        try:
            X_test_global, y_test_global = load_global_test_data()
            if X_test_global is None:
                 print("Arrêt de la simulation car les données de test globales n'ont pas pu être chargées.")
                 return self.metrics
                 
            self.initialize_clients()
            print("Clients initialisés.")

            for round_num in range(1, self.num_rounds + 1):
                round_start_time = time.time()
                print(f"\n--- Round {round_num}/{self.num_rounds} --- ({self.__class__.__name__} {self.encryption})")
                selected_clients = self.select_clients(round_num)
                print(f"Clients sélectionnés: {selected_clients}")
                client_weights = {}
                valid_clients_in_round = []
                round_errors = []

                for client_id in selected_clients:
                    try:
                        print(f"Traitement du client {client_id}...")
                        X_train, y_train = load_client_train_data(client_id)
                        if X_train is None or y_train is None:
                            print(f"  Erreur de chargement des données pour le client {client_id}. Skipping.")
                            round_errors.append(f"Client {client_id}: Data loading error")
                            continue
                        
                        if len(np.unique(y_train)) < 2:
                             print(f"  Pas assez de classes ({len(np.unique(y_train))}) pour le client {client_id}. Skipping training.")
                             round_errors.append(f"Client {client_id}: Not enough classes ({len(np.unique(y_train))})")
                             continue
                             
                        print(f"  Entraînement du client {client_id}...")
                        training_time = self.train_client(client_id, X_train, y_train)
                        print(f"  Entraînement terminé pour le client {client_id} en {training_time:.2f}s.")
                        
                        if hasattr(self.client_models[client_id], 'coef_'):
                             client_weights[client_id] = len(X_train)
                             valid_clients_in_round.append(client_id)
                        else:
                             print(f"  Modèle non ajusté pour le client {client_id} après tentative d'entraînement.")
                             round_errors.append(f"Client {client_id}: Model not fitted after training attempt")
                    except Exception as e:
                         print(f"  Erreur inattendue lors du traitement du client {client_id}: {e}")
                         round_errors.append(f"Client {client_id}: Unexpected error - {e}")
                         import traceback
                         traceback.print_exc() # Log traceback for debugging

                print(f"Clients valides pour l'agrégation: {valid_clients_in_round}")
                if valid_clients_in_round:
                    try:
                        print("Agrégation des modèles...")
                        self.aggregate_models(valid_clients_in_round, client_weights)
                        print("Agrégation terminée.")
                    except Exception as e:
                         print(f"  Erreur lors de l'agrégation des modèles: {e}")
                         round_errors.append(f"Aggregation error: {e}")
                         import traceback
                         traceback.print_exc()
                else:
                     print(f"Round {round_num}: Aucun modèle client valide à agréger.")
                     round_errors.append("No valid clients for aggregation")

                print("Évaluation du modèle global...")
                if self.global_model is not None:
                    try:
                        eval_metrics = self.evaluate_model(X_test_global, y_test_global)
                        print(f"Évaluation terminée: Acc={eval_metrics['accuracy']:.4f}, Loss={eval_metrics['loss']:.4f}")
                    except Exception as e:
                         print(f"  Erreur lors de l'évaluation du modèle global: {e}")
                         eval_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': np.inf}
                         round_errors.append(f"Global evaluation error: {e}")
                         import traceback
                         traceback.print_exc()
                else:
                     print("Modèle global non disponible pour l'évaluation.")
                     eval_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': np.inf}
                     round_errors.append("Global model not available for evaluation")
                     
                comm_cost = self.calculate_communication_cost(valid_clients_in_round)
                memory_usage = self.calculate_memory_usage(valid_clients_in_round)
                entropy = self.calculate_entropy()
                convergence_time = time.time() - round_start_time
                model_size = self.get_model_size()

                # Enregistrer les métriques
                self.metrics['accuracy'].append(eval_metrics['accuracy'])
                self.metrics['precision'].append(eval_metrics['precision'])
                self.metrics['recall'].append(eval_metrics['recall'])
                self.metrics['f1'].append(eval_metrics['f1'])
                self.metrics['loss'].append(eval_metrics['loss'])
                self.metrics['communication_cost'].append(comm_cost)
                self.metrics['memory_usage'].append(memory_usage)
                self.metrics['entropy'].append(entropy)
                self.metrics['convergence_time'].append(convergence_time)
                self.metrics['model_size'].append(model_size)
                # Store round errors if needed
                # self.metrics.setdefault('round_errors', []).append(round_errors)

                print(f"Fin du Round {round_num}. Temps: {convergence_time:.2f}s. "
                      f"Acc: {eval_metrics['accuracy']:.4f}, Loss: {eval_metrics['loss']:.4f}, F1: {eval_metrics['f1']:.4f}")
                if round_errors:
                     print(f"  Erreurs ce round: {len(round_errors)}") # Log count of errors

        except Exception as simulation_error:
             print(f"\n!!! ERREUR FATALE DANS LA SIMULATION {self.__class__.__name__} {self.encryption}: {simulation_error} !!!")
             import traceback
             traceback.print_exc()
        finally:
             # Ensure results are saved even if the simulation loop breaks
             print(f"\nFin de la simulation {self.__class__.__name__} {self.encryption}. Sauvegarde des résultats...")
             self.save_results()
             print("Sauvegarde terminée.")
             return self.metrics

    def save_results(self):
        """Enregistre les résultats de la simulation"""
        # Calculate average memory usage safely
        valid_memory_usage = [m for m in self.metrics['memory_usage'] if m is not None]
        avg_memory_usage = sum(valid_memory_usage) / len(valid_memory_usage) if valid_memory_usage else 0
             
        # Ensure metrics lists are not empty before accessing last element
        final_metrics_dict = {}
        for key, value_list in self.metrics.items():
             if key in ['communication_cost', 'convergence_time']:
                 final_metrics_dict[key] = sum(value_list) if value_list else 0
             elif key == 'memory_usage':
                 final_metrics_dict[key] = avg_memory_usage
             else:
                 final_metrics_dict[key] = value_list[-1] if value_list else None
                 
        self.results = { # Store results in an instance variable
            'algorithm': self.__class__.__name__,
            'encryption': self.encryption,
            'num_clients': self.num_clients,
            'clients_per_round': self.clients_per_round,
            'num_rounds': self.num_rounds,
            'metrics': self.metrics,
            'final_metrics': final_metrics_dict
        }

        filename = f"{OUTPUT_DIR}{self.__class__.__name__}_{self.encryption}.json"
        try:
            with open(filename, 'w') as f:
                # Convert numpy types to standard python types for JSON serialization
                serializable_results = json.loads(json.dumps(self.results, default=lambda x: x.item() if isinstance(x, np.generic) else x.__dict__ if hasattr(x, '__dict__') else str(x)))
                json.dump(serializable_results, f, indent=4)
            print(f"Résultats enregistrés dans {filename}")
        except Exception as e:
             print(f"Error saving results to {filename}: {e}")

class FedMed(FedAvg):
    """Implémentation de FedMed (Federated Median)"""
    def aggregate_models(self, selected_clients, client_weights):
        if self.global_model is None:
            self.global_model = initialize_logistic_regression(NUM_FEATURES, HAR_CLASSES)

        # Collect coefficients and intercepts from fitted models with correct shape
        fitted_coefs = []
        fitted_intercepts = []
        for client_id in selected_clients:
             if hasattr(self.client_models[client_id], 'coef_') and self.client_models[client_id].coef_.shape == self.global_model.coef_.shape:
                 fitted_coefs.append(self.client_models[client_id].coef_)
                 fitted_intercepts.append(self.client_models[client_id].intercept_)
             else:
                  print(f"Warning (FedMed): Client {client_id} model not fitted or shape mismatch, skipping.")

        if not fitted_coefs:
            print("Warning: No fitted models to aggregate in FedMed.")
            return
            
        all_coefs = np.array(fitted_coefs)
        all_intercepts = np.array(fitted_intercepts)

        # Calculate the median for each parameter
        median_coef = np.median(all_coefs, axis=0)
        median_intercept = np.median(all_intercepts, axis=0)

        self.global_model.coef_ = median_coef
        self.global_model.intercept_ = median_intercept

class FedProx(FedAvg):
    """Implémentation de FedProx"""
    def __init__(self, num_clients, clients_per_round, num_rounds, encryption='HE', mu=0.01):
        super().__init__(num_clients, clients_per_round, num_rounds, encryption)
        self.mu = mu

    def train_client(self, client_id, X_train, y_train):
        start_time = time.time()
        
        # Ensure the client model is initialized with correct classes before fitting
        if not hasattr(self.client_models[client_id], 'classes_'):
             self.client_models[client_id].classes_ = np.array(HAR_CLASSES)
             
        # Check for sufficient classes
        unique_classes_in_data = np.unique(y_train)
        if len(unique_classes_in_data) < 2:
            print(f"Warning (FedProx): Client {client_id} training data has only {len(unique_classes_in_data)} class(es). Skipping fit.")
            return time.time() - start_time

        # Standard training
        try:
            self.client_models[client_id].fit(X_train, y_train)
        except ValueError as e:
             print(f"Error fitting client {client_id} model (FedProx): {e}.")
             return time.time() - start_time

        # Apply proximal term adjustment *after* standard fit
        # Ensure global model exists and shapes match before applying prox term
        if (self.global_model is not None and hasattr(self.global_model, 'coef_') and 
            hasattr(self.client_models[client_id], 'coef_') and 
            self.global_model.coef_.shape == self.client_models[client_id].coef_.shape):
            
            # Calculate the difference (gradient w.r.t. prox term)
            diff_coef = self.client_models[client_id].coef_ - self.global_model.coef_
            diff_intercept = self.client_models[client_id].intercept_ - self.global_model.intercept_
            
            # Apply the proximal update (like a gradient step)
            # This is a simplification; true FedProx modifies the local objective function
            self.client_models[client_id].coef_ -= self.mu * diff_coef
            self.client_models[client_id].intercept_ -= self.mu * diff_intercept
        elif self.global_model is None:
             print(f"Warning (FedProx): Global model not available for client {client_id} prox term.")
        elif not hasattr(self.client_models[client_id], 'coef_'):
             print(f"Warning (FedProx): Client {client_id} model not fitted for prox term.")
        # No warning if shapes mismatch, handled in aggregation

        return time.time() - start_time

class FedOpt(FedAvg):
    """Implémentation de FedOpt (using Adam-like server optimizer)"""
    def __init__(self, num_clients, clients_per_round, num_rounds, encryption='HE', learning_rate=0.01, beta1=0.9, beta2=0.999):
        super().__init__(num_clients, clients_per_round, num_rounds, encryption)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def aggregate_models(self, selected_clients, client_weights):
        if self.global_model is None:
            self.global_model = initialize_logistic_regression(NUM_FEATURES, HAR_CLASSES)
            # Initialize Adam moments
            self.m = {'coef': np.zeros_like(self.global_model.coef_), 'intercept': np.zeros_like(self.global_model.intercept_)}
            self.v = {'coef': np.zeros_like(self.global_model.coef_), 'intercept': np.zeros_like(self.global_model.intercept_)}
            self.t = 0

        # Calculate weighted average of client model *updates* (delta from global)
        delta_coef_avg = np.zeros_like(self.global_model.coef_)
        delta_intercept_avg = np.zeros_like(self.global_model.intercept_)

        total_weight = sum(client_weights.values())
        if total_weight == 0:
            print("Warning: Total weight is zero during FedOpt aggregation.")
            return

        num_aggregated = 0
        for client_id in selected_clients:
            if hasattr(self.client_models[client_id], 'coef_') and self.client_models[client_id].coef_.shape == self.global_model.coef_.shape:
                weight = client_weights[client_id] / total_weight
                # Calculate delta: client_model - global_model
                delta_coef = self.client_models[client_id].coef_ - self.global_model.coef_
                delta_intercept = self.client_models[client_id].intercept_ - self.global_model.intercept_
                delta_coef_avg += weight * delta_coef
                delta_intercept_avg += weight * delta_intercept
                num_aggregated += 1
            else:
                 print(f"Warning (FedOpt): Client {client_id} model not fitted or shape mismatch, skipping.")

        if num_aggregated == 0:
            print("Warning: No models were aggregated in FedOpt this round.")
            return

        # Adam optimizer update steps for the *server* (global model)
        self.t += 1
        # Update moments (using the negative delta as the 'gradient')
        grad_coef = -delta_coef_avg 
        grad_intercept = -delta_intercept_avg

        self.m['coef'] = self.beta1 * self.m['coef'] + (1 - self.beta1) * grad_coef
        self.m['intercept'] = self.beta1 * self.m['intercept'] + (1 - self.beta1) * grad_intercept

        self.v['coef'] = self.beta2 * self.v['coef'] + (1 - self.beta2) * (grad_coef ** 2)
        self.v['intercept'] = self.beta2 * self.v['intercept'] + (1 - self.beta2) * (grad_intercept ** 2)

        # Bias correction
        m_hat_coef = self.m['coef'] / (1 - self.beta1 ** self.t)
        m_hat_intercept = self.m['intercept'] / (1 - self.beta1 ** self.t)

        v_hat_coef = self.v['coef'] / (1 - self.beta2 ** self.t)
        v_hat_intercept = self.v['intercept'] / (1 - self.beta2 ** self.t)

        # Update global model parameters
        self.global_model.coef_ -= self.learning_rate * m_hat_coef / (np.sqrt(v_hat_coef) + 1e-8)
        self.global_model.intercept_ -= self.learning_rate * m_hat_intercept / (np.sqrt(v_hat_intercept) + 1e-8)

class CFL_KBS(FedAvg):
    """Implémentation de CFL-KBS"""
    def __init__(self, num_clients, clients_per_round, num_rounds, encryption='HE', num_clusters=4, topk=3):
        super().__init__(num_clients, clients_per_round, num_rounds, encryption='adaptive')
        self.num_clusters = num_clusters
        self.topk = topk
        self.client_clusters = {}
        self.cluster_models = {}
        self.client_characteristics = {}
        self.client_encryption = {}
        self.metrics['encryption_switch_rate'] = []
        self.metrics['strategy_stability'] = []

    def initialize_clients(self):
        super().initialize_clients()
        for client_id in range(1, self.num_clients + 1):
            self.client_characteristics[client_id] = {
                'computational_power': random.uniform(0.1, 1.0),
                'data_sensitivity': random.uniform(0.1, 1.0)
            }
            self.client_encryption[client_id] = self.select_encryption_scheme(client_id)
        self.cluster_clients()

    def select_encryption_scheme(self, client_id):
        comp_power = self.client_characteristics[client_id]['computational_power']
        data_sensitivity = self.client_characteristics[client_id]['data_sensitivity']
        scores = {}
        for scheme, properties in ENCRYPTION_SCHEMES.items():
            security_weight = data_sensitivity
            efficiency_weight = 1.0 - comp_power
            security_score = properties['security_level'] * security_weight
            efficiency_score = (1.0 - properties['computation_cost']) * efficiency_weight
            scores[scheme] = security_score + efficiency_score
        # Simplified: Sort by score and take top-k, then choose the best
        topk_schemes = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)[:self.topk]
        return topk_schemes[0] if topk_schemes else 'HE'

    def cluster_clients(self):
        features = []
        client_ids_list = []
        for client_id, chars in self.client_characteristics.items():
            features.append([chars['computational_power'], chars['data_sensitivity']])
            client_ids_list.append(client_id)
        features = np.array(features)

        try:
            from sklearn.cluster import KMeans
            # Ensure n_clusters is not greater than the number of samples
            actual_n_clusters = min(self.num_clusters, len(features))
            if actual_n_clusters < 1:
                 print("Warning: Not enough clients to form clusters.")
                 self.client_clusters = {0: client_ids_list}
                 self.num_clusters = 1 # Adjust num_clusters if needed
            else:
                 kmeans = KMeans(n_clusters=actual_n_clusters, random_state=SEED, n_init=10)
                 cluster_labels = kmeans.fit_predict(features)
                 self.num_clusters = actual_n_clusters # Update num_clusters if reduced
                 self.client_clusters = {i: [] for i in range(self.num_clusters)}
                 for i, client_id in enumerate(client_ids_list):
                     self.client_clusters[cluster_labels[i]].append(client_id)
        except ImportError:
            # Fallback simple clustering (adjust for num_clusters)
            print("KMeans not available, using simple quantile clustering.")
            q_comp = min(self.num_clusters // 2 if self.num_clusters > 1 else 1, len(features)-1)
            q_sens = min(2, len(features)-1)
            if q_comp < 1 or q_sens < 1:
                 cluster_labels = np.zeros(len(features), dtype=int)
                 self.num_clusters = 1
            else:
                 comp_power_q = pd.qcut(features[:, 0], q=q_comp, labels=False, duplicates='drop')
                 data_sens_q = pd.qcut(features[:, 1], q=q_sens, labels=False, duplicates='drop')
                 cluster_labels = comp_power_q * q_sens + data_sens_q
                 self.num_clusters = len(np.unique(cluster_labels))
                 
            self.client_clusters = {i: [] for i in range(self.num_clusters)}
            for i, client_id in enumerate(client_ids_list):
                 self.client_clusters[cluster_labels[i]].append(client_id)

        # Initialize cluster models
        self.cluster_models = {}
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = initialize_logistic_regression(NUM_FEATURES, HAR_CLASSES)

    # --- select_clients, train_client (mostly same, uses adaptive encryption cost) ---
    def select_clients(self, round_num):
        # Stratified sampling logic (remains the same as previous version)
        selected_clients = []
        clients_available = list(range(1, self.num_clients + 1))
        target_count = self.clients_per_round
        cluster_targets = {}
        total_clients_in_clusters = sum(len(c) for c in self.client_clusters.values())
        if total_clients_in_clusters == 0: 
             return random.sample(clients_available, min(target_count, len(clients_available)))
        for cluster_id, clients in self.client_clusters.items():
            if clients:
                proportion = len(clients) / total_clients_in_clusters
                cluster_targets[cluster_id] = max(1, int(round(proportion * target_count)))
            else: cluster_targets[cluster_id] = 0
        current_target_sum = sum(cluster_targets.values())
        diff = target_count - current_target_sum
        sorted_clusters = sorted(self.client_clusters.keys(), key=lambda cid: len(self.client_clusters[cid]), reverse=True)
        for i in range(abs(diff)):
             if not sorted_clusters: break
             cluster_to_adjust = sorted_clusters[i % len(sorted_clusters)]
             if diff > 0: cluster_targets[cluster_to_adjust] += 1
             elif cluster_targets[cluster_to_adjust] > 1: cluster_targets[cluster_to_adjust] -= 1
        for cluster_id, target in cluster_targets.items():
            clients_in_cluster = self.client_clusters.get(cluster_id, [])
            n_to_select = min(target, len(clients_in_cluster))
            if n_to_select > 0:
                selected = random.sample(clients_in_cluster, n_to_select)
                selected_clients.extend(selected)
        if len(selected_clients) < target_count:
            remaining_available = [c for c in clients_available if c not in selected_clients]
            needed = target_count - len(selected_clients)
            if remaining_available: selected_clients.extend(random.sample(remaining_available, min(needed, len(remaining_available))))
        elif len(selected_clients) > target_count:
             selected_clients = random.sample(selected_clients, target_count)
        return selected_clients
        
    def train_client(self, client_id, X_train, y_train):
        # Ensure the client model is initialized with correct classes before fitting
        if not hasattr(self.client_models[client_id], 'classes_'):
             self.client_models[client_id].classes_ = np.array(HAR_CLASSES)
             
        # Check for sufficient classes
        unique_classes_in_data = np.unique(y_train)
        if len(unique_classes_in_data) < 2:
            print(f"Warning (CFL): Client {client_id} training data has only {len(unique_classes_in_data)} class(es). Skipping fit.")
            return 0 # Return 0 simulated time if skipped
            
        current_encryption = self.client_encryption.get(client_id, 'HE')
        encryption_factor = ENCRYPTION_SCHEMES[current_encryption]['computation_cost']
        start_time = time.time()
        try:
            self.client_models[client_id].fit(X_train, y_train)
        except ValueError as e:
             print(f"Error fitting client {client_id} model (CFL): {e}.")
             return 0 # Return 0 simulated time if error
             
        base_time = time.time() - start_time
        adjusted_time = base_time * (1 + encryption_factor)
        return adjusted_time
        
    def aggregate_models(self, selected_clients, client_weights):
        # Aggregate within clusters
        for cluster_id, cluster_clients_list in self.client_clusters.items():
            selected_cluster_clients = [c for c in selected_clients if c in cluster_clients_list]
            if selected_cluster_clients:
                cluster_client_weights = {c: client_weights[c] for c in selected_cluster_clients if c in client_weights}
                total_cluster_weight = sum(cluster_client_weights.values())
                if total_cluster_weight == 0: continue

                coef_sum = np.zeros_like(self.cluster_models[cluster_id].coef_)
                intercept_sum = np.zeros_like(self.cluster_models[cluster_id].intercept_)
                num_fitted_in_cluster = 0
                for client_id in selected_cluster_clients:
                    if hasattr(self.client_models[client_id], 'coef_') and self.client_models[client_id].coef_.shape == self.cluster_models[cluster_id].coef_.shape:
                        weight = cluster_client_weights[client_id] / total_cluster_weight
                        coef_sum += weight * self.client_models[client_id].coef_
                        intercept_sum += weight * self.client_models[client_id].intercept_
                        num_fitted_in_cluster += 1
                if num_fitted_in_cluster > 0:
                    self.cluster_models[cluster_id].coef_ = coef_sum
                    self.cluster_models[cluster_id].intercept_ = intercept_sum

        # Aggregate cluster models to global model
        if self.global_model is None:
            self.global_model = initialize_logistic_regression(NUM_FEATURES, HAR_CLASSES)

        cluster_total_clients = {cid: len(c_list) for cid, c_list in self.client_clusters.items()}
        total_weight_all_clusters = sum(cluster_total_clients.values())
        if total_weight_all_clusters == 0: return

        global_coef = np.zeros_like(self.global_model.coef_)
        global_intercept = np.zeros_like(self.global_model.intercept_)
        num_aggregated_clusters = 0 # Keep track of how many clusters contributed
        for cluster_id, weight in cluster_total_clients.items():
            # Check if cluster model exists, is fitted, and has the correct shape
            if (cluster_id in self.cluster_models and 
                hasattr(self.cluster_models[cluster_id], 'coef_') and 
                self.cluster_models[cluster_id].coef_.shape == self.global_model.coef_.shape):
                
                normalized_weight = weight / total_weight_all_clusters
                global_coef += normalized_weight * self.cluster_models[cluster_id].coef_
                global_intercept += normalized_weight * self.cluster_models[cluster_id].intercept_
                num_aggregated_clusters += 1
            else:
                 print(f"Warning (CFL Global Agg): Cluster {cluster_id} model not fitted or shape mismatch, skipping.")

        # Only update global model if at least one cluster model contributed
        if num_aggregated_clusters > 0:
            self.global_model.coef_ = global_coef
            self.global_model.intercept_ = global_intercept
        else:
             print("Warning (CFL Global Agg): No valid cluster models were aggregated to the global model.")

    # --- calculate_communication_cost, memory_usage, entropy (same as before) ---
    def calculate_communication_cost(self, selected_clients):
        total_cost = 0
        for client_id in selected_clients:
            encryption = self.client_encryption.get(client_id, 'HE')
            client_cost = 0.1 * ENCRYPTION_SCHEMES[encryption]['communication_cost']
            total_cost += client_cost
        return total_cost

    def calculate_memory_usage(self, selected_clients):
        total_usage = 0
        if not selected_clients: return 0
        for client_id in selected_clients:
            encryption = self.client_encryption.get(client_id, 'HE')
            client_usage = 0.1 * ENCRYPTION_SCHEMES[encryption]['memory_usage']
            total_usage += client_usage
        # Return average usage for the round
        return total_usage / len(selected_clients) 

    def calculate_entropy(self):
        total_entropy = 0
        if not self.client_encryption: return 0
        for client_id, encryption in self.client_encryption.items():
            total_entropy += ENCRYPTION_SCHEMES[encryption]['entropy']
        return total_entropy / len(self.client_encryption)
        
    def calculate_encryption_switch_rate(self, old_encryptions, current_round_client_encryptions):
        """Calculates switch rate based ONLY on clients selected in the current round."""
        switches = 0
        selected_client_ids = current_round_client_encryptions.keys()
        if not selected_client_ids:
            return 0
        for client_id in selected_client_ids:
            if client_id in old_encryptions and old_encryptions[client_id] != current_round_client_encryptions[client_id]:
                switches += 1
        return switches / len(selected_client_ids)

    def calculate_strategy_stability(self):
        """Calculates stability based on encryption consistency within clusters."""
        cluster_consistency_scores = []
        total_clients_in_calc = 0
        for cluster_id, clients in self.client_clusters.items():
            if not clients: continue
            encryption_counts = defaultdict(int)
            for client_id in clients:
                enc = self.client_encryption.get(client_id, 'HE')
                encryption_counts[enc] += 1
            if encryption_counts:
                max_count = max(encryption_counts.values())
                consistency = max_count / len(clients)
                cluster_consistency_scores.append(consistency * len(clients))
                total_clients_in_calc += len(clients)
        if total_clients_in_calc == 0: return 0.0
        return sum(cluster_consistency_scores) / total_clients_in_calc

    def run_simulation(self):
        print(f"Démarrage de la simulation {self.__class__.__name__} avec {self.num_clusters} clusters et Top-{self.topk}")
        X_test_global, y_test_global = load_global_test_data()
        if X_test_global is None: return self.metrics
        self.initialize_clients()

        for round_num in range(1, self.num_rounds + 1):
            start_time = time.time()
            old_encryptions = self.client_encryption.copy()
            selected_clients = self.select_clients(round_num)
            
            # Update encryption for selected clients for this round
            current_round_client_encryptions = {}
            for client_id in selected_clients:
                 new_encryption = self.select_encryption_scheme(client_id)
                 self.client_encryption[client_id] = new_encryption # Update the main dict
                 current_round_client_encryptions[client_id] = new_encryption
                 
            switch_rate = self.calculate_encryption_switch_rate(old_encryptions, current_round_client_encryptions)
            self.metrics['encryption_switch_rate'].append(switch_rate)
            stability = self.calculate_strategy_stability()
            self.metrics['strategy_stability'].append(stability)

            client_weights = {}
            valid_clients_in_round = []
            for client_id in selected_clients:
                X_train, y_train = load_client_train_data(client_id)
                if X_train is None or y_train is None or len(np.unique(y_train)) < 2:
                     print(f"Skipping client {client_id} training in CFL (data/class issue).")
                     continue
                     
                training_time = self.train_client(client_id, X_train, y_train)
                if hasattr(self.client_models[client_id], 'coef_'):
                     client_weights[client_id] = len(X_train)
                     valid_clients_in_round.append(client_id)

            # Re-cluster periodically
            if round_num % 5 == 0:
                print(f"Round {round_num}: Re-clustering clients...")
                self.cluster_clients()

            if valid_clients_in_round:
                self.aggregate_models(valid_clients_in_round, client_weights)
            else:
                 print(f"Round {round_num} (CFL): No valid client models to aggregate.")

            if self.global_model is not None:
                eval_metrics = self.evaluate_model(X_test_global, y_test_global)
            else:
                 eval_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': np.inf}
                 
            comm_cost = self.calculate_communication_cost(valid_clients_in_round)
            memory_usage = self.calculate_memory_usage(valid_clients_in_round)
            entropy = self.calculate_entropy()
            convergence_time = time.time() - start_time
            model_size = self.get_model_size()

            # Append metrics
            self.metrics['accuracy'].append(eval_metrics['accuracy'])
            self.metrics['precision'].append(eval_metrics['precision'])
            self.metrics['recall'].append(eval_metrics['recall'])
            self.metrics['f1'].append(eval_metrics['f1'])
            self.metrics['loss'].append(eval_metrics['loss'])
            self.metrics['communication_cost'].append(comm_cost)
            self.metrics['memory_usage'].append(memory_usage)
            self.metrics['entropy'].append(entropy)
            self.metrics['convergence_time'].append(convergence_time)
            self.metrics['model_size'].append(model_size)
            # Switch rate and stability already appended

            if round_num % 5 == 0 or round_num == 1:
                print(f"Round {round_num}/{self.num_rounds} - "
                      f"Acc: {eval_metrics['accuracy']:.4f}, Loss: {eval_metrics['loss']:.4f}, F1: {eval_metrics['f1']:.4f}, "
                      f"Switch: {switch_rate:.4f}, Stability: {stability:.4f}")

        self.save_results()
        return self.metrics

    def save_results(self):
        """Enregistre les résultats de la simulation pour CFL-KBS"""
        valid_memory_usage = [m for m in self.metrics['memory_usage'] if m is not None]
        avg_memory_usage = sum(valid_memory_usage) / len(valid_memory_usage) if valid_memory_usage else 0
        valid_switch_rate = [r for r in self.metrics['encryption_switch_rate'] if r is not None]
        avg_switch_rate = sum(valid_switch_rate) / len(valid_switch_rate) if valid_switch_rate else 0
        valid_stability = [s for s in self.metrics['strategy_stability'] if s is not None]
        final_stability = valid_stability[-1] if valid_stability else 0
        
        final_metrics_dict = {}
        for key, value_list in self.metrics.items():
             if key in ['communication_cost', 'convergence_time']:
                 final_metrics_dict[key] = sum(value_list) if value_list else 0
             elif key == 'memory_usage':
                 final_metrics_dict[key] = avg_memory_usage
             elif key == 'encryption_switch_rate':
                 final_metrics_dict['avg_encryption_switch_rate'] = avg_switch_rate
             elif key == 'strategy_stability':
                 final_metrics_dict['final_strategy_stability'] = final_stability
             else:
                 final_metrics_dict[key] = value_list[-1] if value_list else None
                 
        self.results = {
            'algorithm': self.__class__.__name__,
            'encryption': 'adaptive',
            'num_clients': self.num_clients,
            'clients_per_round': self.clients_per_round,
            'num_rounds': self.num_rounds,
            'num_clusters': self.num_clusters,
            'topk': self.topk,
            'metrics': self.metrics,
            'final_metrics': final_metrics_dict,
            'client_clusters': {str(k): v for k, v in self.client_clusters.items()},
            'client_encryption_final': {str(k): v for k, v in self.client_encryption.items()}
        }
        filename = f"{OUTPUT_DIR}{self.__class__.__name__}_clusters{self.num_clusters}_topk{self.topk}.json"
        try:
            with open(filename, 'w') as f:
                serializable_results = json.loads(json.dumps(self.results, default=lambda x: x.item() if isinstance(x, np.generic) else x.__dict__ if hasattr(x, '__dict__') else str(x)))
                json.dump(serializable_results, f, indent=4)
            print(f"Résultats enregistrés dans {filename}")
        except Exception as e:
             print(f"Error saving CFL results to {filename}: {e}")

# --- Fonctions utilitaires pour charger les données (avec gestion d'erreur) ---
def load_client_train_data(client_id):
    file_path = os.path.join(DATA_DIR, f"client_{client_id}_data.csv") # Changed file name pattern
    try:
        data = pd.read_csv(file_path)
        if data.empty:
             print(f"Warning: Training data file for client {client_id} is empty.")
             return None, None
        y = data["label"] # Changed label column name
        X = data.drop(columns=["label"]) # Changed label column name
        # Ensure X has the correct number of features
        if X.shape[1] != NUM_FEATURES:
             print(f"Warning: Client {client_id} data has {X.shape[1]} features, expected {NUM_FEATURES}. Skipping.")
             return None, None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except FileNotFoundError:
        print(f"Error: Training data file not found for client {client_id} at {file_path}")
        return None, None
    except Exception as e:
         print(f"Error loading training data for client {client_id}: {e}")
         return None, None

# def load_client_test_data(client_id):
#     file_path = os.path.join(DATA_DIR, f"client_{client_id}_test.csv")
#     try:
#         data = pd.read_csv(file_path)
#         if data.empty:
#              print(f"Warning: Test data file for client {client_id} is empty.")
#              return None, None
#         y = data['activity_id']
#         X = data.drop(columns=['activity_id'])
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#         return X_scaled, y
#     except FileNotFoundError:
#         print(f"Error: Test data file not found for client {client_id} at {file_path}")
#         return None, None
#     except Exception as e:
#          print(f"Error loading test data for client {client_id}: {e}")
#          return None, None

def load_global_test_data():
    file_path = os.path.join(DATA_DIR, "global_test_data.csv")
    try:
        data = pd.read_csv(file_path)
        if data.empty:
             print(f"Warning: Global test data file is empty at {file_path}.")
             return None, None
        y = data["label"] # Changed label column name
        X = data.drop(columns=["label"]) # Changed label column name
        # Ensure X has the correct number of features
        if X.shape[1] != NUM_FEATURES:
             print(f"Warning: Global test data has {X.shape[1]} features, expected {NUM_FEATURES}.")
             return None, None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Global test data loaded successfully from {file_path}. Shape: {X_scaled.shape}")
        return X_scaled, y
    except FileNotFoundError:
        print(f"Error: Global test data file not found at {file_path}")
        return None, None
    except Exception as e:
         print(f"Error loading global test data: {e}")
         return None, None

# --- Fonction principale pour exécuter toutes les simulations ---
def run_all_simulations():
    algorithms = [
        # {"name": "FedAvg", "class": FedAvg},
        # {"name": "FedMed", "class": FedMed},
        {"name": "FedProx", "class": FedProx},
        # {"name": "FedOpt", "class": FedOpt},
        # {"name": "CFL_KBS", "class": CFL_KBS}
    ]
    # encryptions = [\'AES\', \'RSA\', \'ECC\', \'HE\', \'SMPC\', \'DP\', \'PE\'] # Not needed for CFL-only run
    results = []

    for algo in algorithms:
        # if algo["name"] == "CFL_KBS":
        #     for num_clusters in [4]: # Only 4 clusters
        #         for topk in [2]: # Only Top-2
        #             print(f"\n{'='*50}\nExécution de {algo['name']} avec {num_clusters} clusters et Top-{topk}\n{'='*50}")
        #             try:
        #                 model = algo["class"](NUM_CLIENTS, CLIENTS_PER_ROUND, NUM_ROUNDS, num_clusters=num_clusters, topk=topk)
        #                 model.run_simulation()
        #                 # Safely access results after run
        #                 if hasattr(model, 'results') and 'final_metrics' in model.results:
        #                      final_metrics = model.results['final_metrics']
        #                      results.append({
        #                          "algorithm": algo["name"], "encryption": "adaptive", "num_clusters": num_clusters, "topk": topk,
        #                          **final_metrics # Unpack final metrics dict
        #                      })
        #                 else:
        #                      print(f"Results or final_metrics not found for {algo['name']} run.")
        #             except Exception as e:
        #                  print(f"Error running {algo['name']} with clusters={num_clusters}, topk={topk}: {e}")
        #                  import traceback
        #                  traceback.print_exc() # Print full traceback
        # else: # Run non-CFL algorithms
        # if algo["name"] == "FedAvg": # Ensure we only run FedAvg now
        # if algo["name"] == "FedMed": # Ensure we only run FedMed now
        if algo["name"] == "FedProx": # Ensure we only run FedProx now
            for encryption in ["HE"]: # Only HE
                print(f"\n{'='*50}\nExécution de {algo['name']} avec chiffrement {encryption}\n{'='*50}")
                try:
                    model = algo["class"](NUM_CLIENTS, CLIENTS_PER_ROUND, NUM_ROUNDS, encryption=encryption)
                    model.run_simulation()
                    if hasattr(model, 'results') and 'final_metrics' in model.results:
                         final_metrics = model.results['final_metrics']
                         results.append({
                             "algorithm": algo["name"], "encryption": encryption,
                             **final_metrics
                         })
                    else:
                         print(f"Results or final_metrics not found for {algo['name']} {encryption} run.")
                except Exception as e:
                     print(f"Error running {algo['name']} with encryption {encryption}: {e}")
                     import traceback
                     traceback.print_exc()

    results_df = pd.DataFrame(results)
    # Add missing columns if they don\'t exist from certain runs (e.g., CFL specific metrics)
    for col in ["num_clusters", "topk", "avg_encryption_switch_rate", "final_strategy_stability"]:
         if col not in results_df.columns:
             results_df[col] = None
             
    results_df.to_csv(f"{OUTPUT_DIR}simulation_summary_cfl_only.csv", index=False) # Save to a specific file
    print("\nSimulations CFL-KBS terminées. Résumé enregistré dans simulation_summary_cfl_only.csv")
    return results_df
if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    results_df = run_all_simulations()
    print("\nRésumé des résultats finaux:")
    display_cols = [
        "algorithm", "encryption", "num_clusters", "topk", "accuracy", "f1", "loss",
        "communication_cost", "memory_usage", "model_size", "convergence_time", 
        "avg_encryption_switch_rate", "final_strategy_stability"
    ]
    display_cols = [col for col in display_cols if col in results_df.columns]
    # Format floats for better readability
    float_cols = results_df.select_dtypes(include=["float"]).columns
    format_dict = {col: ".4f".format for col in float_cols}
    print(results_df[display_cols].to_string(formatters=format_dict))
