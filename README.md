# predictPPI

This study introduces MFPIC (Multi-Feature Protein Interaction Classifier), a novel computational model for predicting protein-protein interactions (PPIs). The model integrates enhanced sequence-based features, including Spaced Conjoint Triad (SCT) and Amino Acid Pairwise Distance (AAPD), along with established methods such as Position-Specific Scoring Matrices (PSSM) and AAindex features. MFPIC captures complex sequence motifs and spatial relationships within proteins, improving the accuracy of PPI predictions. Evaluated on Saccharomyces cerevisiae, Helicobacter pylori, and Human datasets, the model outperforms state-of-the-art methods, demonstrating its potential for biological and therapeutic research​.

Data; is the folder where the database files are located. The AAindex file is a database containing amino acid physicochemical properties, substitution matrices and statistical protein contact potentials. It can be accessed as an open source from the https://www.genome.jp/aaindex/ website.

The seqs file lists protein sequences specific to the biological organism.

The pairs file contains interaction data specific to the biological organism.

The MFPIC file contains the model.

Protein-Protein Interaction Prediction
This project uses a deep learning model to predict protein-protein interactions (PPI). Below is an explanation of each part of the code.

Requirements
Python 3.x
NumPy
Pandas
Scikit-learn
TensorFlow
Data Loading
interactions = pd.read_csv("interactions.csv")
sequences = pd.read_csv("seqsHel2.csv")
aaindex = pd.read_csv("aaindex.csv", header=None)
Data is loaded from interactions.csv, seqsHel2.csv, and aaindex.csv files.

Amino Acid Index Verification
expected_rows = 20
if len(aaindex) != expected_rows:
    raise ValueError(f"Expected {expected_rows} rows in aaindex.csv, but found {len(aaindex)}")
The aaindex.csv file is verified to contain 20 rows.

Mapping Amino Acids
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aaindex_dict = {aa: aaindex.iloc[idx].values for idx, aa in enumerate(amino_acids)}
Amino acids are mapped to the aaindex DataFrame.

Feature Extraction
Conjoint Triad Calculation
def get_conjoint_triads(sequence):
    triads = []
    for i in range(len(sequence) - 2):
        triad = sequence[i:i+3]
        triads.append(triad)
    return triads
Conjoint Triad Calculation with Gaps
def get_conjoint_triads_with_gaps(sequence):
    triads_with_gaps = []
    for i in range(len(sequence) - 4):
        triad = sequence[i] + sequence[i+2] + sequence[i+4]
        triads_with_gaps.append(triad)
    return triads_with_gaps
Feature Extraction Function
def extract_features(sequences, triads):
    triad_counts = []
    for sequence in sequences:
        triads_in_sequence = get_conjoint_triads(sequence)
        triads_in_sequence_with_gaps = get_conjoint_triads_with_gaps(sequence)
        triad_count = [triads_in_sequence.count(triad) for triad in triads]
        triad_count_with_gaps = [triads_in_sequence_with_gaps.count(triad) for triad in triads]
        triad_counts.append(triad_count + triad_count_with_gaps)
    return triad_counts
Data Merging and Cleaning
data = interactions.merge(sequences, left_on="protein1", right_on="protein_code", how="left").drop("protein_code", axis=1)
data = data.merge(sequences, left_on="protein2", right_on="protein_code", suffixes=("_1", "_2"), how="left").drop("protein_code", axis=1)
data = data.dropna(subset=["sequence_1", "sequence_2"])
Data is merged and rows with missing sequences are dropped.

Feature Extraction and Target Variable
X_triad_1 = extract_features(data['sequence_1'], unique_triads)
X_triad_2 = extract_features(data['sequence_2'], unique_triads)
X = np.hstack([X_triad_1, X_triad_2])
y = data['interaction'].values
Data Splitting and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Data is split into training and test sets and scaled.

Model Definition and Training
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

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr, model_checkpoint])
Model Evaluation
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred_binary)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'MCC Score: {mcc:.4f}')
The model is evaluated on the test set and various performance metrics are calculated.
