import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Verileri yükleme
interactions = pd.read_csv("interactions.csv")
sequences = pd.read_csv("seqsHel2.csv")
aaindex = pd.read_csv("aaindex.csv", header=None)

# AAindex kontrol
if len(aaindex) != 20:
    raise ValueError("AAindex veri seti 20 amino asidi temsil etmelidir.")

# Hidropati skorları
hydropathy_scores = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
                     'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
                     'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}

# Fonksiyonlar
def calculate_hydropathy(sequence):
    """Bir sekansın hidropati skorunu hesaplar."""
    return np.mean([hydropathy_scores[aa] for aa in sequence if aa in hydropathy_scores])

def calculate_aapd(sequence):
    """Bir sekansın amino asit çiftleri arasındaki mesafeyi hesaplar."""
    positions = {aa: [] for aa in set(sequence)}
    for i, aa in enumerate(sequence):
        positions[aa].append(i)
    distances = []
    for aa, pos_list in positions.items():
        for i in range(len(pos_list) - 1):
            distances.append(pos_list[i + 1] - pos_list[i])
    if len(distances) == 0:
        return 0  # Hiçbir çift yoksa varsayılan değer
    return np.mean(distances)

def extract_features(sequences):
    """SCT, AAPD ve hydropathy özelliklerini çıkarır."""
    sct_features = []
    aapd_features = []
    hydropathy_features = []

    for seq in sequences:
        # SCT (dummy olarak sekans uzunluğu kullanılıyor)
        sct_features.append(len(seq))
        # AAPD
        aapd_features.append(calculate_aapd(seq))
        # Hidropati
        hydropathy_features.append(calculate_hydropathy(seq))

    return np.array(sct_features).reshape(-1, 1), np.array(aapd_features).reshape(-1, 1), np.array(hydropathy_features).reshape(-1, 1)

# Veri birleştirme ve özellik çıkarımı
data = interactions.merge(sequences, left_on="protein1", right_on="protein_code", how="left").drop("protein_code", axis=1)
data = data.merge(sequences, left_on="protein2", right_on="protein_code", suffixes=("_1", "_2"), how="left").drop("protein_code", axis=1)

# SCT, AAPD ve hydropathy özelliklerini çıkarma
X_sct_1, X_aapd_1, X_hydropathy_1 = extract_features(data['sequence_1'])
X_sct_2, X_aapd_2, X_hydropathy_2 = extract_features(data['sequence_2'])

# Tüm özellikleri birleştirme
X = np.hstack([X_sct_1, X_sct_2, X_aapd_1, X_aapd_2, X_hydropathy_1, X_hydropathy_2])
y = data["interaction"].values

# Ablation analizi
def ablation_analysis(X, y, feature_groups):
    results = []
    for group_name, features_to_remove in feature_groups.items():
        X_ablation = np.delete(X, features_to_remove, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_ablation, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Sequential([
            Dense(256, input_dim=X_train.shape[1]),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.5),
            Dense(128),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])

        y_pred = model.predict(X_test) > 0.5
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict(X_test))
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "Removed Features": group_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "MCC": mcc
        })

    return pd.DataFrame(results)

# Özellik grupları (her bir özellik grubu için indeksler)
feature_groups = {
    "SCT": [0, 1],
    "AAPD": [2, 3],
    "Hydropathy": [4, 5],
}

# Ablation analizini çalıştır
results_df = ablation_analysis(X, y, feature_groups)

# Sonuçları kaydet ve yazdır
results_df.to_csv("ablation_results.csv", index=False)
print(results_df)
