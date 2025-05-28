# Multi Layer Perceptron (MLP) model
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize the sizes of the layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        
        # Initialize biases to zero
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Learning rate
        self.learning_rate = learning_rate

    # Rectified Linear Unit (ReLU) activation function
    def relu(self, x):
        return np.maximum(0, x)
        
    # Softmax activation function
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        # Prevent log(0) & log(1)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calculate cross-entropy loss for each sample
        loss = y_true * np.log(y_pred)

        # Sum the loss across classes, applying negative sign
        # to convert to positive loss
        loss = -np.sum(loss, axis=1)

        # Average the loss across all samples
        return np.mean(loss)
    
    def forward(self, X):
        z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.weights_hidden_output) + self.bias_output
        a2 = self.softmax(z2)

        # PROD
        # return a2
        # DEV
        return a1, a2

    def backward(self, X, y_true, a1, a2):
        # Nombre d'exemples
        m = X.shape[0]

        # Calcul du gradient de la sortie (softmax + cross-entropy)
        dz2 = a2 - y_true  # (batch, output_size)
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Gradient pour la couche cachée
        da1 = np.dot(dz2, self.weights_hidden_output.T)
        dz1 = da1 * (a1 > 0)  # dérivée de ReLU
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Mise à jour des poids et biais
        self.weights_input_hidden   -= self.learning_rate * dw1
        self.bias_hidden            -= self.learning_rate * db1

        self.weights_hidden_output  -= self.learning_rate * dw2
        self.bias_output            -= self.learning_rate * db2

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}...", end='\r')
            # Mélange des données
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]

            # Propagation avant
            a1, a2 = self.forward(X)
            # Calcul de la perte
            loss = self.cross_entropy_loss(y, a2)
            # Rétropropagation
            self.backward(X, y, a1, a2)
            # Affichage de la perte
            #if (epoch + 1) % 1 == 0:
                #print(f"\tÉpoque {epoch+1}/{epochs} - Perte : {loss:.4f}")

    def save_model(self, filename: str):
        np.savez(filename,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                weights_input_hidden=self.weights_input_hidden,
                bias_hidden=self.bias_hidden,
                weights_hidden_output=self.weights_hidden_output,
                bias_output=self.bias_output)
