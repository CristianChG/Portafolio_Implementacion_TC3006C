# Dataset link: https://www.kaggle.com/datasets/madhavtesting/heart-stroke-dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

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

tree = DecisionTreeClassifier(max_depth=19, random_state=42)
tree.fit(X_train, y_train)

y_pred_train = tree.predict(X_train)
print("Accuracy Train:", accuracy_score(y_train, y_pred_train))

y_pred_val = tree.predict(X_val)
print("Accuracy Val:", accuracy_score(y_val, y_pred_val))

y_pred_test = tree.predict(X_test)
print("Accuracy Test:", accuracy_score(y_test, y_pred_test))


plt.figure(figsize=(25, 10))  
plot_tree(
    tree,
    filled=True,              
    feature_names=["bias"] + df.drop("stroke", axis=1).columns.tolist(),
    class_names=["No Stroke", "Stroke"],
    rounded=True,
    fontsize=8
)
plt.show()
