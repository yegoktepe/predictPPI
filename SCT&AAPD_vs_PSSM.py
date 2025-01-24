# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Helper Functions (PSSM, SCT, AAPD, Feature Extraction)
def calculate_pssm(sequences, seq_length):
    amino_acids = "ARNDCEQGHILKMFPSTWYV"
    pssm_matrix = np.zeros((len(amino_acids), seq_length))
    for seq in sequences:
        for i, aa in enumerate(seq):
            if aa in amino_acids:
                pssm_matrix[amino_acids.index(aa), i] += 1
    pssm_matrix = pssm_matrix / len(sequences)
    pssm_matrix[pssm_matrix == 0] = 1e-9
    background_freq = np.ones(len(amino_acids)) / len(amino_acids)
    pssm_matrix = np.log2(pssm_matrix / background_freq[:, None])
    return [np.mean(pssm_matrix[:, :min(len(seq), seq_length)], axis=1) for seq in sequences]

def get_conjoint_triads(sequence, gap=0):
    triads = []
    for i in range(len(sequence) - (2 + gap)):
        triad = sequence[i] + sequence[i + 1 + gap] + sequence[i + 2 + gap]
        triads.append(triad)
    return triads

def calculate_aapd(sequence):
    positions = {aa: [] for aa in set(sequence)}
    for i, aa in enumerate(sequence):
        positions[aa].append(i)
    distances = []
    for pos_list in positions.values():
        distances += [abs(pos_list[i+1] - pos_list[i]) for i in range(len(pos_list) - 1)]
    return np.mean(distances) if distances else 0

# Dataset Loading
interactions = pd.read_csv("interactionsfile.csv")
sequences = pd.read_csv("seqsfile.csv")
aaindex = pd.read_csv("aaindex.csv", header=None)

data = interactions.merge(sequences, left_on="protein1", right_on="protein_code", how="left").merge(
    sequences, left_on="protein2", right_on="protein_code", suffixes=("_1", "_2"), how="left"
).dropna()

# Determine Max Sequence Length for PSSM
max_seq_length = max(data["sequence_1"].apply(len).max(), data["sequence_2"].apply(len).max())

# PSSM Features
X_pssm_1 = calculate_pssm(data['sequence_1'], max_seq_length)
X_pssm_2 = calculate_pssm(data['sequence_2'], max_seq_length)
X_pssm = np.hstack([X_pssm_1, X_pssm_2])

# SCT Features
all_triads = set()
for seq in pd.concat([data['sequence_1'], data['sequence_2']]):
    all_triads.update(get_conjoint_triads(seq))
unique_triads = list(all_triads)
X_sct_1 = np.array([[seq.count(triad) for triad in unique_triads] for seq in data['sequence_1']])
X_sct_2 = np.array([[seq.count(triad) for triad in unique_triads] for seq in data['sequence_2']])
X_sct = np.hstack([X_sct_1, X_sct_2])

# AAPD Features
X_aapd_1 = np.array([calculate_aapd(seq) for seq in data['sequence_1']]).reshape(-1, 1)
X_aapd_2 = np.array([calculate_aapd(seq) for seq in data['sequence_2']]).reshape(-1, 1)
X_aapd = np.hstack([X_aapd_1, X_aapd_2])

# Combine Features
X_combined = np.hstack([X_pssm, X_sct, X_aapd])
y = data['interaction'].values

# Split Data
X_train_pssm, X_test_pssm, y_train, y_test = train_test_split(X_pssm, y, test_size=0.2, random_state=42)
X_train_combined, X_test_combined, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Scale Features
scaler_pssm = StandardScaler()
X_train_pssm = scaler_pssm.fit_transform(X_train_pssm)
X_test_pssm = scaler_pssm.transform(X_test_pssm)

scaler_combined = StandardScaler()
X_train_combined = scaler_combined.fit_transform(X_train_combined)
X_test_combined = scaler_combined.transform(X_test_combined)

# Model Definition
def build_model(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        Dense(128),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train Models
model_pssm = build_model(X_train_pssm.shape[1])
model_combined = build_model(X_train_combined.shape[1])

model_pssm.fit(X_train_pssm, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
model_combined.fit(X_train_combined, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

# Evaluate Models
y_pred_pssm = (model_pssm.predict(X_test_pssm) > 0.5).astype(int)
y_pred_combined = (model_combined.predict(X_test_combined) > 0.5).astype(int)

metrics_pssm = {
    "Accuracy": accuracy_score(y_test, y_pred_pssm),
    "Precision": precision_score(y_test, y_pred_pssm),
    "Recall": recall_score(y_test, y_pred_pssm),
    "F1": f1_score(y_test, y_pred_pssm),
    "MCC": matthews_corrcoef(y_test, y_pred_pssm),
}

metrics_combined = {
    "Accuracy": accuracy_score(y_test, y_pred_combined),
    "Precision": precision_score(y_test, y_pred_combined),
    "Recall": recall_score(y_test, y_pred_combined),
    "F1": f1_score(y_test, y_pred_combined),
    "MCC": matthews_corrcoef(y_test, y_pred_combined),
}

# Print Results
print("PSSM Only Model Metrics:", metrics_pssm)
print("PSSM + SCT + AAPD Model Metrics:", metrics_combined)
