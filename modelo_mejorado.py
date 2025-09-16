# Dataset link: https://www.kaggle.com/datasets/madhavtesting/heart-stroke-dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
from etl import cargar_datos

X_train, y_train, X_val, y_val, X_test, y_test = cargar_datos(
    ruta_csv='healthcare-dataset-stroke.csv',
    analisis=False,
    bias=False
)

# Modelo base
tree = DecisionTreeClassifier(max_depth=15, random_state=42)
tree.fit(X_train, y_train)

# Modelo con poda de complejidad
path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  

train_scores_f1_ccp, val_scores_f1_ccp = [], []

for alpha in ccp_alphas:
    model = DecisionTreeClassifier(max_depth=15, ccp_alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    train_scores_f1_ccp.append(f1_score(y_train, model.predict(X_train)))
    val_scores_f1_ccp.append(f1_score(y_val, model.predict(X_val)))

# Gráfica F1-Score vs ccp_alpha
plt.figure(figsize=(8,5))
plt.plot(ccp_alphas, train_scores_f1_ccp, marker='o', label="F1 (Train)")
plt.plot(ccp_alphas, val_scores_f1_ccp, marker='o', label="F1 (Val)")
plt.xlabel("ccp_alpha (poda de complejidad)")
plt.ylabel("F1-Score")
plt.title("Efecto de la Poda de Complejidad en el Árbol")
plt.legend()
plt.show()

# Modelo con alpha bueno
alpha_optimo = 0.0015

tree_podado = DecisionTreeClassifier(max_depth=15, ccp_alpha=alpha_optimo, random_state=42)
tree_podado.fit(X_train, y_train)

print("Modelo podado (ccp_alpha =", alpha_optimo, ") ")
print("Accuracy Train:", accuracy_score(y_train, tree_podado.predict(X_train)))
print("Accuracy Val:", accuracy_score(y_val, tree_podado.predict(X_val)))
print("Accuracy Test:", accuracy_score(y_test, tree_podado.predict(X_test)))
print("F1-Score Train:", f1_score(y_train, tree_podado.predict(X_train)))
print("F1-Score Val:", f1_score(y_val, tree_podado.predict(X_val)))
print("F1-Score Test:", f1_score(y_test, tree_podado.predict(X_test)))