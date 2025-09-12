# Dataset link: https://www.kaggle.com/datasets/madhavtesting/heart-stroke-dataset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class NeuronaLogistica:
    def __init__(self, learning_rate, epocas):
        self.learning_rate = learning_rate
        self.epocas = epocas
        self.pesos = None
        self.historial_costos = []
        self.historial_accuracy = []

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
    def gradiente_descendiente(self, X, y):
        m = len(y)
        n = len(X[0])
        self.pesos = [0.0] * n
        self.historial_costos = []
        self.historial_accuracy = []

        for epoca in range(1, self.epocas+1):
            grad = [0.0] * n

            for i in range(m):
                h = self.funcion_hypotesis(X[i])
                error = h - y[i]
                for j in range(n):
                    grad[j] += error * X[i][j]

            for j in range(n):
                grad[j] /= m
                self.pesos[j] -= self.learning_rate * grad[j]

            costo = self.cross_entropy(X, y)
            acc = self.calcular_accuracy(X, y)
            self.historial_costos.append(costo)
            self.historial_accuracy.append(acc)

            if epoca % 200 == 0 or epoca == 1 or epoca == self.epocas:
                print(f"Epoca {epoca}/{self.epocas} Costo: {costo:.2f} Accuracy: {acc*100:.2f}%")

    # Clasificación
    def clasificar(self, X):
        predicciones = []
        for fila in X:
            p = self.funcion_hypotesis(fila)
            predicciones.append(1 if p >= 0.5 else 0) 
        return predicciones


# ETL
df = pd.read_csv("healthcare-dataset-stroke.csv")
print(df.describe())
print("Valores nulos por columna:")
print(df.isnull().sum())

# Imputación avanzada BMI por género y grupos de edad
df["age_group"] = pd.cut(df["age"], bins=[0,18,30,40,50,60,70,80,100])
df["bmi"] = (
    df.groupby(["gender", "age_group"], observed=True)["bmi"]
      .transform(lambda x: x.fillna(x.median()))
)
df.drop("age_group", axis=1, inplace=True)

# Categóricas a numéricas
df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype("int64")
df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0}).astype("int64")
df = pd.get_dummies(
    df,
    columns=["work_type", "Residence_type", "smoking_status"],
    drop_first=True,
    dtype=int
)

# Escalado Z-score
variables_numericas = ["age", "avg_glucose_level", "bmi"]
df[variables_numericas] = (df[variables_numericas] - df[variables_numericas].mean()) / df[variables_numericas].std()

# Matriz de correlación 
corr_matrix = df.corr()

# Valores de la correlación
print("\nMatriz de correlación:")
print(corr_matrix)

plt.figure(figsize=(14,12))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, center=0)
plt.title("Matriz de correlación")
plt.show()

# Eliminar columnas innecesarias
columnas_inecesarias = [
    'work_type_children', # Multicolinealidad con edad
    'ever_married',    # Multicolinealidad con edad
    'work_type_Never_worked', # Muy baja correlación
]

df.drop(columnas_inecesarias, axis=1, inplace=True)

# X, y y bias
X = df.drop("stroke", axis=1).astype(float).values
y = df["stroke"].astype(int).values
X = np.c_[np.ones(X.shape[0]), X]  # bias

print("Shape final de X:", X.shape)
print("Shape final de y:", y.shape)
print(df.head())

# Partición de datos (60% train, 20% val, 20% test)
np.random.seed(42)
indices = np.arange(X.shape[0]); np.random.shuffle(indices)
X = X[indices]; y = y[indices]

train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val   = X[train_size:train_size + val_size]
y_val   = y[train_size:train_size + val_size]
X_test  = X[train_size + val_size:]
y_test  = y[train_size + val_size:]

print("Entrenamiento:", X_train.shape, y_train.shape)
print("Validación:",   X_val.shape,   y_val.shape)
print("Prueba:",       X_test.shape,  y_test.shape)

# Entrenar 
modelo = NeuronaLogistica(learning_rate=0.005, epocas=1000)
modelo.gradiente_descendiente(X_train, y_train)

acc_train = modelo.calcular_accuracy(X_train, y_train)
acc_val   = modelo.calcular_accuracy(X_val, y_val)
acc_test  = modelo.calcular_accuracy(X_test, y_test)

print(f"\nExactitud entrenamiento: {acc_train*100:.2f}%")
print(f"Exactitud validación:    {acc_val*100:.2f}%")
print(f"Exactitud prueba:        {acc_test*100:.2f}%")

# Gráficas
plt.figure(figsize=(8,5))
plt.plot(modelo.historial_costos, label="Costo (train)")
plt.xlabel("Épocas"); 
plt.ylabel("Cross-entropy")
plt.show()

plt.figure(figsize=(8,5))
plt.plot([a*100 for a in modelo.historial_accuracy], label="Accuracy (train)")
plt.xlabel("Épocas"); 
plt.ylabel("Accuracy (%)")
plt.show()

ejemplos = X_test[:10]
reales = y_test[:10]
predicciones = modelo.clasificar(ejemplos)

print("\nPredicciones de ejemplo:")
for i in range(len(ejemplos)):
    print(f"Real = {reales[i]}, Predicho = {predicciones[i]}")