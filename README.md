# mnist_mlp

## Description

Le MLP (Multilayer Perceptron) est constitué de trois couches principales :
- La couche d'entrée (Input layer)
- La couche cachée (Hidden layer)
- La couche de sortie (Output layer)

## Input layer
Elle sert à faire entrer dans le MLP chaque pixel de l'image. Dans le cas de MNIST, chaque image est composée de 28x28 pixels, soit 784 pixels au total.

## Hidden layer
Dans cette couche, on réalise le produit scalaire des pixels d'entrée avec les poids associés à cette couche. Puis, on applique une fonction d'activation **ReLU** (Rectified Linear Unit) pour introduire de la non-linéarité. 
La fonction ReLU produit des valeurs dans l'intervalle [0, ∞].

## Output layer
Dans la couche de sortie, on réalise à nouveau un produit scalaire des valeurs issues de la couche cachée et des poids de cette couche. Ensuite, on applique **Softmax** pour obtenir une probabilité pour chaque classe. Cette probabilité est calculée de manière dépendante, c’est-à-dire que la somme des probabilités est égale à 1.

Exemple de sortie :
```python
[0.65, 0.15, 0.10, ...]
```
Cela signifie que :
- La probabilité d'appartenir à la classe 0 est de **65%**,
- La probabilité d'appartenir à la classe 1 est de **15%**,
- La probabilité d'appartenir à la classe 2 est de **10%**.

## Loss
La fonction **cross_entropy_loss** est utilisée pour mesurer la **distance** entre la réponse prédite (yp) et la vérité (y). La cross-entropy calcule l'écart entre la distribution des probabilités prédites par le modèle et la distribution des probabilités de la vérité terrain (qui est un vecteur one-hot).

L’objectif de l’apprentissage du MLP est de **minimiser cette distance** pour améliorer les prédictions à chaque itération.
