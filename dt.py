# Dataset link: https://www.kaggle.com/datasets/madhavtesting/heart-stroke-dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import plot_tree
from etl import cargar_datos


# Cargar datos
X_train, y_train, X_val, y_val, X_test, y_test = cargar_datos(ruta_csv = 'healthcare-dataset-stroke.csv',analisis=False, bias=False)

tree = DecisionTreeClassifier(max_depth=15, random_state=42)
tree.fit(X_train, y_train)

y_pred_train = tree.predict(X_train)
print("Accuracy Train:", accuracy_score(y_train, y_pred_train))

y_pred_val = tree.predict(X_val)
print("Accuracy Val:", accuracy_score(y_val, y_pred_val))

y_pred_test = tree.predict(X_test)
print("Accuracy Test:", accuracy_score(y_test, y_pred_test))

print("F1-Score train:", f1_score(y_train, y_pred_train))
print("F1-Score val:", f1_score(y_val, y_pred_val))
print("F1-Score test:", f1_score(y_test, y_pred_test))

# Visualización del árbol
plt.figure(figsize=(25, 10))  
plot_tree(
    tree,
    filled=True,
    class_names=["No Stroke", "Stroke"],
    rounded=True,
    fontsize=8
)
plt.show()

param_range = range(1, 16)

train_scores_acc = []
val_scores_acc = []
train_scores_f1 = []
val_scores_f1 = []

# Evaluar diferentes profundidades del árbol
for depth in param_range:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar en train
    y_train_pred = model.predict(X_train)
    train_scores_acc.append(accuracy_score(y_train, y_train_pred))
    train_scores_f1.append(f1_score(y_train, y_train_pred))
    
    # Evaluar en val
    y_val_pred = model.predict(X_val)
    val_scores_acc.append(accuracy_score(y_val, y_val_pred))
    val_scores_f1.append(f1_score(y_val, y_val_pred))

# Gráfica Accuracy
plt.figure(figsize=(8,5))
plt.plot(param_range, train_scores_acc, label="Accuracy (Train)")
plt.plot(param_range, val_scores_acc, label="Accuracy (Val)")
plt.xlabel("Profundidad máxima del árbol")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Profundidad del Árbol")
plt.legend()
plt.show()

# Gráfica F1-Score
plt.figure(figsize=(8,5))
plt.plot(param_range, train_scores_f1, label="F1-Score (Train)")
plt.plot(param_range, val_scores_f1, label="F1-Score (Val)")
plt.xlabel("Profundidad máxima del árbol")
plt.ylabel("F1-Score")
plt.title("F1-Score vs Profundidad del Árbol")
plt.legend()
plt.show()