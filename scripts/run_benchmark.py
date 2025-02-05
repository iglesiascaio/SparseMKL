import numpy as np
import pandas as pd
from MKLpy.algorithms import EasyMKL
from sklearn.metrics import accuracy_score
from MKLpy.preprocessing import normalization
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time


# Function to load the Adult dataset
def get_adult_data(frac=0.33, data_dir="../data"):
    file_path = f"{data_dir}/adult.csv"
    df = pd.read_csv(file_path)

    # Encode labels as +1 and -1
    df["income_binary"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else -1)
    labels = df["income_binary"].values

    # Select numerical features
    features = df[
        ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    ].values

    # Sample a fraction of the dataset
    df_sampled = df.sample(frac=frac, random_state=456)
    X = df_sampled[
        ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    ].values
    y = df_sampled["income_binary"].values

    # Normalize data
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Convert to float32 for memory efficiency
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Train-test split (80-20)
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Reshape labels to column vectors
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, y_train, X_test, y_test


# Load data
X_train, y_train, X_test, y_test = get_adult_data(frac=0.33)

# Define kernel specifications
kernels = [
    {"type": "linear"},
    {"type": "poly", "degree": 3, "coef0": 1.0},
    {"type": "poly", "degree": 2, "coef0": 1.0},
    {"type": "rbf", "gamma": 0.5},
    {"type": "rbf", "gamma": 0.3},
    {"type": "rbf", "gamma": 0.1},
]


# Compute kernel matrices
def compute_kernels(X_train, X_test, kernels):
    from MKLpy.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel

    kernel_functions = {
        "linear": linear_kernel,
        "poly": polynomial_kernel,
        "rbf": rbf_kernel,
    }

    K_train_list = []
    K_test_list = []

    for kernel in kernels:
        kernel_type = kernel["type"]
        params = kernel.copy()
        params.pop("type")

        K_train = kernel_functions[kernel_type](X_train, X_train, **params)
        K_test = kernel_functions[kernel_type](X_train, X_test, **params)

        K_train_list.append(np.array(K_train))
        K_test_list.append(np.array(K_test))

    return K_train_list, K_test_list


# Compute kernels
start_time = time.time()
K_train_list, K_test_list = compute_kernels(X_train, X_test, kernels)
print(f"Kernel computation time: {time.time() - start_time:.2f} seconds")

# Debug: Print kernel and label shapes
print("Train Kernel Shapes:", [K.shape for K in K_train_list])
print("Test Kernel Shapes:", [K.shape for K in K_test_list])
print("Train Labels Shape:", y_train.shape)
print("Test Labels Shape:", y_test.shape)

# Train an interpretable sparse MKL model
λ = 0.5
start_time = time.time()
mkl = EasyMKL(lam=λ)
mkl.fit(K_train_list, y_train.ravel())  # Flatten labels for compatibility
β = mkl.solution.weights  # Use the correct attribute name
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

import ipdb

ipdb.set_trace()

# Predict using combined kernel
y_pred_train = mkl.predict(K_train_list)
# y_pred_test = mkl.predict(K_test_list)
y_pred_test = mkl.predict([K.T for K in K_test_list])  # Transpose test kernels

# Compute accuracy
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)


print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
print(f"Test Accuracy: {accuracy_test * 100:.2f}%")
print(f"Training time: {time.time() - start_time:.2f} seconds")
print(f"Beta (Kernel Weights): {β}")


# Function to print confusion matrix and metrics
def print_confusion_metrics(y_actual, y_pred, set_name="Data Set"):
    cm = confusion_matrix(y_actual, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_actual, y_pred, pos_label=1)
    recall = recall_score(y_actual, y_pred, pos_label=1)
    f1 = f1_score(y_actual, y_pred, pos_label=1)

    print(f"--------------- {set_name} ---------------")
    print("Confusion Matrix:")
    print(f"            Predicted")
    print(f"            -1     +1")
    print(f"Actual -1   {tn:<6} {fp:<6}")
    print(f"       +1   {fn:<6} {tp:<6}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("")


# Print confusion matrices
print_confusion_metrics(y_train, y_pred_train, "Train Set")
print_confusion_metrics(y_test, y_pred_test, "Test Set")
