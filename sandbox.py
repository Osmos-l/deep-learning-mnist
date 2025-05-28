from models.mlp import MLP
import numpy as np

import kagglehub
import numpy as np

import time

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print(path)

def load_idx_images(path):
    with open(path,'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[16:].reshape(-1, 28*28) / 255.0

def load_idx_labels(path):
    with open(path,'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data[8:]

# Load full training set
X_full = load_idx_images(f'{path}/train-images.idx3-ubyte')
y_full = load_idx_labels(f'{path}/train-labels.idx1-ubyte')
# Load test set
X_test = load_idx_images(f'{path}/t10k-images.idx3-ubyte')
y_test = load_idx_labels(f'{path}/t10k-labels.idx1-ubyte')

# One-hot encoding des labels
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_full_oh = one_hot(y_full)
y_test_oh = one_hot(y_test)

# Initialisation du MLP
input_size = 28 * 28
hidden_size = 128  # à ajuster selon vos besoins
output_size = 10
learning_rate = 0.1
nb_epochs = 2000

# Réinitialisation du MLP à chaque test pour comparer équitablement
mlp = MLP(input_size, hidden_size, output_size, learning_rate)

print(f"\nEntrainement sur {X_full.shape[0]} exemples pendant {nb_epochs} époques.")
start = time.time()
mlp.train(X_full, y_full_oh, epochs=nb_epochs)
end = time.time()

# Prédiction sur le test set
_, y_pred_test = mlp.forward(X_test)
y_pred_labels = np.argmax(y_pred_test, axis=1)

# Affichage de la précision
accuracy = np.mean(y_pred_labels == y_test)
print(f"Précision sur {X_test.shape[0]} exemples du test set après {nb_epochs} époques : {accuracy:.2%}")
print(f"Temps d'entraînement pour {nb_epochs} époques : {end - start:.2f} secondes")

mlp.save_model("model.npz")