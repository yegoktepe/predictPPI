# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load data
interactions = pd.read_csv("interactionsfile.csv")
sequences = pd.read_csv("seqsfile.csv")
aaindex = pd.read_csv("aaindex.csv", header=None)

# Verify the number of rows in aaindex
expected_rows = 20  # There are 20 standard amino acids
if len(aaindex) != expected_rows:
    raise ValueError(f"Expected {expected_rows} rows in aaindex.csv, but found {len(aaindex)}")

# Mapping of amino acids to their respective rows in the aaindex DataFrame
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aaindex_dict = {aa: aaindex.iloc[idx].values for idx, aa in enumerate(amino_acids)}

# Function to calculate conjoint triads (including the new version with gaps)
def get_conjoint_triads(sequence):
    triads = []
    for i in range(len(sequence) - 2):
        triad = sequence[i:i+3]
        triads.append(triad)
    return triads

# Function to calculate modified conjoint triads with gaps
def get_conjoint_triads_with_gaps(sequence):
    triads_with_gaps = []
    for i in range(len(sequence) - 4):
        triad = sequence[i] + sequence[i+2] + sequence[i+4]
        triads_with_gaps.append(triad)
    return triads_with_gaps

# Function to extract features
def extract_features(sequences, triads):
    triad_counts = []
    for sequence in sequences:
        triads_in_sequence = get_conjoint_triads(sequence)
        triads_in_sequence_with_gaps = get_conjoint_triads_with_gaps(sequence)
        triad_count = [triads_in_sequence.count(triad) for triad in triads]
        triad_count_with_gaps = [triads_in_sequence_with_gaps.count(triad) for triad in triads]
        triad_counts.append(triad_count + triad_count_with_gaps)
    return triad_counts

# Merge data
data = interactions.merge(sequences, left_on="protein1", right_on="protein_code", how="left").drop("protein_code", axis=1)
data = data.merge(sequences, left_on="protein2", right_on="protein_code", suffixes=("_1", "_2"), how="left").drop("protein_code", axis=1)

# Drop rows with missing sequences
data = data.dropna(subset=["sequence_1", "sequence_2"])

# Extract unique triads
all_triads = set()
for sequence in data["sequence_1"]:
    all_triads.update(get_conjoint_triads(sequence))
for sequence in data["sequence_2"]:
    all_triads.update(get_conjoint_triads(sequence))
unique_triads = list(all_triads)

# Feature extraction
X_triad_1 = extract_features(data['sequence_1'], unique_triads)
X_triad_2 = extract_features(data['sequence_2'], unique_triads)

# Combine features
X = np.hstack([X_triad_1, X_triad_2])

# Target variable
y = data['interaction'].values

# Check shapes of X and y
print(f"X shape before: {np.array(X).shape}")
print(f"y shape before: {y.shape}")

# Ensure X and y have consistent sample sizes
min_samples = min(len(X), len(y))
X = np.array(X)[:min_samples]
y = y[:min_samples]

# Verify shapes after adjustment
print(f"X shape after: {X.shape}")
print(f"y shape after: {y.shape}")

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model definition
model = Sequential([
    Dense(512, input_dim=X_train.shape[1]),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.5),
    Dense(256),
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

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Model training
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Model evaluation
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred_binary)

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'MCC Score: {mcc:.4f}')
