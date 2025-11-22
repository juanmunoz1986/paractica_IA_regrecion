import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# ==========================================
# PASO 1: LIMPIEZA Y PREPARACIÓN DE DATOS
# ==========================================

# 1. Cargar datos
df = pd.read_csv('Mental Health Dataset.csv')

# 2. SELECCIÓN DE COLUMNAS MEJORADA
# Hemos elegido las 5 variables más potentes para predecir
columnas_a_usar = [
    'treatment',              # OBJETIVO (Y)
    'family_history',         # Variable 1
    'Growing_Stress',         # Variable 2
    'Mental_Health_History',  # Variable 3 (NUEVA: Muy potente)
    'care_options',           # Variable 4 (NUEVA: Acceso a ayuda)
    'Social_Weakness'         # Variable 5 (NUEVA: Aislamiento)
]
df = df[columnas_a_usar]

# 3. LIMPIEZA DE HUECOS (NaN)
df = df.fillna('No')

# 4. DICCIONARIO DE TRADUCCIÓN
traductor = {
    'Yes': 1,
    'No': 0,
    'Maybe': 0.5,      # Le damos un peso medio a la duda
    'Not sure': 0.5,
    "Don't know": 0,   # Si no sabe de opciones, es como no tenerlas
    'High': 1,
    'Medium': 1,
    'Low': 0
}

# 5. APLICAR TRADUCCIÓN
for col in df.columns:
    df[col] = df[col].replace(traductor)

# Verificación rápida
print("--- Muestra de datos listos ---")
print(df.head())

# ==========================================
# PASO 2: ENTRENAMIENTO DEL MODELO
# ==========================================

# 1. Definir X e y
X = df.drop('treatment', axis=1)
y = df['treatment']

# 2. Dividir en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Configurar y Entrenar
laRegresion = LogisticRegression(max_iter=1000)
laRegresion.fit(X_train, y_train)

# ==========================================
# PASO 3: LA MATEMÁTICA (LOS COEFICIENTES)
# ==========================================
print("\n--- LA FÓRMULA SECRETA DEL MODELO ---")
coef = laRegresion.coef_[0]
intercepto = laRegresion.intercept_[0]

print(f"Intercepto (Base): {intercepto:.4f}")
print("Pesos descubiertos:")
variables = X.columns
for i in range(len(variables)):
    print(f" - {variables[i]}: {coef[i]:.4f}")

# ==========================================
# PASO 4: SISTEMA EXPERTO INTERACTIVO
# ==========================================
print("\n-SISTEMA DE DIAGNÓSTICO DE IA ---")
print("Responde las preguntas para calcular tu probabilidad.")

# Input manual respetando el orden de las columnas
v1_familia = float(input("1. ¿Historial familiar de depresión? (1=Sí, 0=No): "))
v2_estres  = float(input("2. ¿Siente estrés creciente? (1=Sí, 0=No): "))
v3_personal= float(input("3. ¿Historial PERSONAL previo? (1=Sí, 0=No, 0.5=Quizás): "))
v4_cuidado = float(input("4. ¿Conoce sus opciones de cuidado médico? (1=Sí, 0=No, 0.5=No seguro): "))
v5_social  = float(input("5. ¿Siente debilidad social/aislamiento? (1=Sí, 0=No): "))

# Cálculo manual de la fórmula (z)
z = intercepto + (coef[0]*v1_familia) + (coef[1]*v2_estres) + (coef[2]*v3_personal) + (coef[3]*v4_cuidado) + (coef[4]*v5_social)

# Función Sigmoide
probabilidad = 1 / (1 + np.exp(-z))

print(f"\n--- RESULTADO DEL DIAGNÓSTICO ---")
print(f"Probabilidad de necesitar tratamiento: {probabilidad*100:.2f}%")

if probabilidad >= 0.5:
    print("CONCLUSIÓN: SE RECOMIENDA buscar ayuda profesional.")
else:
    print("CONCLUSIÓN: Riesgo bajo por el momento.")

# ==========================================
# PASO 5: EVALUACIÓN VISUAL (MATRIZ)
# ==========================================

# Predicciones
y_pred = laRegresion.predict(X_test)

# Matriz
matrixC = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrixC, display_labels=['No Tratar', 'Sí Tratar'])

# Gráfica - Crear figura y ejes explícitamente para evitar ventanas duplicadas
fig, ax = plt.subplots(figsize=(8, 6))
cm_display.plot(ax=ax, cmap='viridis')
ax.set_title("Matriz de Confusión: Modelo Mejorado")
plt.show()



# ==========================================
# PASO 6: CÁLCULO DE MÉTRICAS FINALES
# ==========================================

# Asegurarnos de tener las predicciones listas
y_pred = laRegresion.predict(X_test)

print("\n--- REPORTE DE CALIDAD DEL MODELO ---")

# 1. Accuracy (Exactitud)
# ¿Qué porcentaje total de veces acertó?
exactitud = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy (Exactitud):    {exactitud:.4f} ({exactitud*100:.2f}%)")

# 2. Precision (Precisión)
# De los que el modelo dijo "SÍ necesita ayuda", ¿cuántos eran verdad?
# Evita falsas alarmas.
precision = metrics.precision_score(y_test, y_pred)
print(f"Precision (Precisión):   {precision:.4f} ({precision*100:.2f}%)")

# 3. Recall (Exhaustividad/Sensibilidad) - ¡VITAL EN SALUD!
# De toda la gente que REALMENTE necesitaba ayuda, ¿a cuántos encontró el modelo?
# Queremos que esto sea alto para no dejar a nadie enfermo sin atender.
recall = metrics.recall_score(y_test, y_pred)
print(f"Recall (Sensibilidad):   {recall:.4f} ({recall*100:.2f}%)")

# 4. F1 Score
# Es el promedio entre Precisión y Recall. Es la nota final justa.
f1 = metrics.f1_score(y_test, y_pred)
print(f"F1 Score (Nota Final):   {f1:.4f}")