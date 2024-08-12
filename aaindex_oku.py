import numpy as np

def parse_aaindex1(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Aminoasit sırası
    amino_acids_order = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ]
    num_features = 566
    num_amino_acids = len(amino_acids_order)
    
    # Özellik matrisi
    feature_matrix = np.zeros((num_features, num_amino_acids))
    feature_index = -1
    reading_values = False
    
    for line in lines:
        if line.startswith('H '):  # Yeni bir özellik başlıyor
            feature_index += 1
        elif line.startswith('I '):  # Aminoasit isimleri
            reading_values = True
            value_lines = []
        elif reading_values:
            value_lines.append(line.strip().split())
            if len(value_lines) == 2:
                values = np.array(value_lines, dtype=float)
                values = values.flatten()
                feature_matrix[feature_index, :] = values
                reading_values = False
        elif line.startswith('//'):  # Özelliğin sonu
            continue
    
    return feature_matrix.T

# AAindex1 dosyasının yolunu belirleyin
file_path = 'aaindex1.txt'

# AAindex1 dosyasını okuyup matrisi oluşturun
feature_matrix = parse_aaindex1(file_path)

np.savetxt("matris.csv", feature_matrix, delimiter=",", fmt="%d")

print("Feature matrix shape:", feature_matrix.shape)
print("First feature for all amino acids:", feature_matrix[:, 0])
