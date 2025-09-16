import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cargar_datos(ruta_csv = 'healthcare-dataset-stroke.csv', analisis = False, bias = False):
    df = pd.read_csv(ruta_csv)
    if analisis:
        print(df.describe())
        print("Valores nulos por columna:")
        print(df.isnull().sum())
        
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
    
    if analisis:
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
    
    if bias:
        X = np.c_[np.ones(X.shape[0]), X]  # bias
    
    if analisis:
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
    
    if analisis:
        print("Entrenamiento:", X_train.shape, y_train.shape)
        print("Validación:",   X_val.shape,   y_val.shape)
        print("Prueba:",       X_test.shape,  y_test.shape)
        
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = cargar_datos(ruta_csv = 'healthcare-dataset-stroke.csv',analisis=False, bias=True)
