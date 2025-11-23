import os

# --- CONFIGURACIÓN PARA SILENCIAR TENSORFLOW ---
# 0 = todos los mensajes (por defecto)
# 1 = filtrar INFO
# 2 = filtrar INFO y WARNING (Lo recomendado para limpiar consola)
# 3 = filtrar todo menos ERRORES críticos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf


# ---------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------
print("Cargando dataset...") # Print de control para saber que corre
try:
    df = pd.read_csv('Mental Health Dataset.csv')
except FileNotFoundError:
    print("\nError: No se encuentra el archivo 'Mental Health Dataset.csv'.")
    print("Asegúrate de que el archivo está en la misma carpeta que tu script 'neu.py'.")
    exit()

# ---------------------------------------------------------
# 2. LIMPIEZA Y PREPROCESAMIENTO
# ---------------------------------------------------------
# Verificamos si existe la columna antes de borrarla para evitar errores
if 'Timestamp' in df.columns:
    df = df.drop('Timestamp', axis=1)

df['self_employed'] = df['self_employed'].fillna('No')

# Tu regla: United States = 1, Otros = 0
df['Country'] = np.where(df['Country'] == 'United States', 1, 0)

# Target: Yes -> 1, No -> 0
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

# Binarias
cols_yes_no = ['self_employed', 'family_history', 'Growing_Stress', 
               'Changes_Habits', 'Mental_Health_History', 'Coping_Struggles', 
               'Work_Interest', 'Social_Weakness']

for col in cols_yes_no:
    # Verificar si la columna existe para evitar errores
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Maybe': 0.5, 'Not sure': 0.5}).fillna(0)

# One-Hot Encoding
cols_to_encode = ['Gender', 'Occupation', 'Days_Indoors', 'Mood_Swings', 
                  'mental_health_interview', 'care_options']
# Filtramos solo las columnas que realmente existen en el df
cols_present = [c for c in cols_to_encode if c in df.columns]
df = pd.get_dummies(df, columns=cols_present, drop_first=True)

# ---------------------------------------------------------
# 3. PREPARACIÓN DE DATOS
# ---------------------------------------------------------
X = df.drop('treatment', axis=1).values
y = df['treatment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. ENTRENAMIENTO CON DROPOUT (Mejor generalización)
# ---------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    
    # Capa 1
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Apaga el 20% de neuronas al azar (evita memorización)
    
    # Capa 2
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2), # Apaga otro 20%
    
    # Salida
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Usamos EarlyStopping para que no pierdas tiempo
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Vigilamos la pérdida en validación
    patience=10,        # Le damos un poco más de paciencia
    restore_best_weights=True
)

print("Entrenando con Dropout...")

# IMPORTANTE: validation_split=0.1 ayuda al EarlyStopping a ser más preciso
history = model.fit(
    X_train, 
    y_train, 
    epochs=30, 
    batch_size=32, 
    verbose=1, 
    validation_split=0.1, # Usamos un 10% extra para validar mientras entrena
    callbacks=[early_stop]
)

# ---------------------------------------------------------
# 5. EVALUACIÓN
# ---------------------------------------------------------
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n" + "="*30)
print("      REPORTE DE MÉTRICAS      ")
print("="*30)

# Calcular todas las métricas
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
error = 1 - acc

# Mostrar métricas
print("\nMatriz de Confusión:")
print(cm)
print("\n" + "-"*30)
print("MÉTRICAS DEL MODELO:")
print("-"*30)
print(f"Error: {error:.2f}")
print(f"Exactitud: {acc:.2f}")
print(f"Precisión: {precision:.2f}")
print(f"Exhaustividad: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("-"*30)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Tratamiento', 'Si Tratamiento']))

# ---------------------------------------------------------
# 6. VISUALIZACIONES
# ---------------------------------------------------------
print("\nGenerando gráficas...")

# 6.1 Gráfica del Error durante el Entrenamiento (simple, como ejemplo)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], color='blue', linewidth=2)
plt.xlabel('Iteración', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Error en función de las Iteraciones', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('error_entrenamiento.png', dpi=300, bbox_inches='tight')
print("-> Gráfica del error guardada en 'error_entrenamiento.png'")
plt.show()

# 6.2 Matriz de Confusión Visualizada
from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(8, 6))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Tratamiento', 'Sí Tratamiento'])
cm_display.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
print("-> Matriz de confusión guardada en 'matriz_confusion.png'")
plt.show()

# ---------------------------------------------------------
# 7. GUARDAR
# ---------------------------------------------------------
print("\n" + "="*30)
print("      GUARDANDO SISTEMA...     ")
print("="*30)

# Guardar modelo
model.save('modelo_salud_mental.keras')
print("-> Modelo guardado: modelo_salud_mental.keras")

# Guardar scaler
joblib.dump(scaler, 'scaler_salud_mental.pkl')
print("-> Scaler guardado: scaler_salud_mental.pkl")

# Guardar columnas
cols_model = df.drop('treatment', axis=1).columns
joblib.dump(cols_model, 'columnas_entrenamiento.pkl')
print("-> Columnas guardadas: columnas_entrenamiento.pkl")

print("\n¡Proceso finalizado!")