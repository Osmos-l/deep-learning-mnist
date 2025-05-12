import kagglehub
import numpy as np

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

print(X_full[0])
print(y_full[1])

import matplotlib.pyplot as plt

# Remodeler X_full[0] en une image 28x28
image = X_full[0].reshape(28, 28)

# Afficher l'image
plt.imshow(image, cmap='gray')
plt.show()