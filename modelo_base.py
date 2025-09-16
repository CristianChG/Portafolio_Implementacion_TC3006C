# Dataset link: https://www.kaggle.com/datasets/madhavtesting/heart-stroke-dataset

import matplotlib.pyplot as plt
import math
from etl import cargar_datos

class NeuronaLogistica:
    def __init__(self, learning_rate, epocas):
        self.learning_rate = learning_rate
        self.epocas = epocas
        self.pesos = None
        self.historial_costos = {"train": [], "val": [], "test": []}
        self.historial_accuracy = {"train": [], "val": [], "test": []}
        

    # Hipótesis
    def funcion_hypotesis(self, x_fila):
        z = 0.0
        for i in range(len(x_fila)):
            z += x_fila[i] * self.pesos[i]
        return 1.0 / (1.0 + math.exp(-z))

    # Cross Entropy
    def cross_entropy(self, X, y):
        m = len(y)
        costo = 0.0
        eps = 1e-8
        for i in range(m):
            h = self.funcion_hypotesis(X[i])
            if h < eps: h = eps
            if h > 1 - eps: h = 1 - eps
            costo += -(y[i]*math.log(h) + (1-y[i])*math.log(1-h))
        return costo/m

    # Accuracy
    def calcular_accuracy(self, X, y):
        correctos = 0
        for i in range(len(y)):
            h = self.funcion_hypotesis(X[i])
            pred = 1 if h >= 0.5 else 0 
            if pred == y[i]:
                correctos += 1
        return correctos / len(y)

    # GD
    def gradiente_descendiente(self, X_train, y_train, X_val, y_val, X_test, y_test):
        m = len(y_train)
        n = len(X_train[0])
        self.pesos = [0.0] * n
        self.historial_costos = {"train": [], "val": [], "test": []}
        self.historial_accuracy = {"train": [], "val": [], "test": []}
        
        for epoca in range(1, self.epocas+1):
            grad = [0.0] * n

            for i in range(m):
                h = self.funcion_hypotesis(X_train[i])
                error = h - y_train[i]
                for j in range(n):
                    grad[j] += error * X_train[i][j]

            for j in range(n):
                grad[j] /= m
                self.pesos[j] -= self.learning_rate * grad[j]
                
            for nombre, X, y in [("train", X_train, y_train),
                                 ("val", X_val, y_val),
                                 ("test", X_test, y_test)]:
                self.historial_costos[nombre].append(self.cross_entropy(X, y))
                self.historial_accuracy[nombre].append(self.calcular_accuracy(X, y))

            if epoca % 200 == 0 or epoca == 1 or epoca == self.epocas:
                print(f"Época {epoca}/{self.epocas} "
                      f"Train Acc: {self.historial_accuracy['train'][-1]*100:.2f}%, "
                      f"Val Acc: {self.historial_accuracy['val'][-1]*100:.2f}%, "
                      f"Test Acc: {self.historial_accuracy['test'][-1]*100:.2f}%")
    # Clasificación
    def clasificar(self, X):
        predicciones = []
        for fila in X:
            p = self.funcion_hypotesis(fila)
            predicciones.append(1 if p >= 0.5 else 0) 
        return predicciones

# Cargar datos
X_train, y_train, X_val, y_val, X_test, y_test = cargar_datos(ruta_csv = 'healthcare-dataset-stroke.csv',analisis=False, bias=True)

# Entrenar 
modelo = NeuronaLogistica(learning_rate=0.005, epocas=1000)
modelo.gradiente_descendiente(X_train, y_train, X_val, y_val, X_test, y_test)

acc_train = modelo.calcular_accuracy(X_train, y_train)
acc_val   = modelo.calcular_accuracy(X_val, y_val)
acc_test  = modelo.calcular_accuracy(X_test, y_test)

print(f"\nExactitud entrenamiento: {acc_train*100:.2f}%")
print(f"Exactitud validación:    {acc_val*100:.2f}%")
print(f"Exactitud prueba:        {acc_test*100:.2f}%")

# Gráficas
plt.figure(figsize=(8,5))
plt.plot(modelo.historial_costos["train"], label="Costo (Train)")
plt.plot(modelo.historial_costos["val"], label="Costo (Val)")
plt.plot(modelo.historial_costos["test"], label="Costo (Test)")
plt.xlabel("Épocas")
plt.ylabel("Cross-entropy")
plt.title("Evolución del costo")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot([a*100 for a in modelo.historial_accuracy["train"]], label="Accuracy (Train)")
plt.plot([a*100 for a in modelo.historial_accuracy["val"]], label="Accuracy (Val)")
plt.plot([a*100 for a in modelo.historial_accuracy["test"]], label="Accuracy (Test)")
plt.xlabel("Épocas")
plt.ylabel("Accuracy (%)")
plt.title("Evolución del accuracy")
plt.legend()
plt.show()

ejemplos = X_test[:10]
reales = y_test[:10]
predicciones = modelo.clasificar(ejemplos)

print("\nPredicciones de ejemplo:")
for i in range(len(ejemplos)):
    print(f"Real = {reales[i]}, Predicho = {predicciones[i]}")