import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# ==========================================
# PARTE 1: ENTRENAMIENTO DEL MODELO (BACKEND)
# ==========================================

# 1. Cargar y Limpiar Datos
df = pd.read_csv('Mental Health Dataset.csv')

# Selección de variables poderosas
columnas_a_usar = [
    'treatment', 
    'family_history', 
    'Growing_Stress', 
    'Mental_Health_History',
    'care_options', 
    'Social_Weakness'
]
df = df[columnas_a_usar]
df = df.fillna('No')

# Traducción a números
traductor = {
    'Yes': 1, 'No': 0, 
    'Maybe': 0.5, 'Not sure': 0.5, "Don't know": 0,
    'High': 1, 'Medium': 1, 'Low': 0
}

for col in df.columns:
    df[col] = df[col].replace(traductor)

# 2. Entrenar Modelo
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

laRegresion = LogisticRegression(max_iter=1000)
laRegresion.fit(X_train, y_train)

# Extraer coeficientes para usarlos en la interfaz
coef = laRegresion.coef_[0]
intercepto = laRegresion.intercept_[0]

# ==========================================
# PARTE 2: INTERFAZ GRÁFICA (FRONTEND)
# ==========================================

def calcular_probabilidad():
    try:
        # 1. Obtener datos de los menús desplegables
        # Convertimos las respuestas de texto (Sí/No) a números (1.0/0.0)
        
        # Variable 1: Historial Familiar
        val_familia = 1.0 if combo_familia.get() == "Sí" else 0.0
        
        # Variable 2: Estrés
        val_estres = 1.0 if combo_estres.get() == "Sí" else 0.0
        
        # Variable 3: Historial Personal
        opcion_personal = combo_personal.get()
        if opcion_personal == "Sí": val_personal = 1.0
        elif opcion_personal == "Quizás": val_personal = 0.5
        else: val_personal = 0.0
        
        # Variable 4: Opciones de Cuidado
        opcion_cuidado = combo_cuidado.get()
        if opcion_cuidado == "Sí": val_cuidado = 1.0
        elif opcion_cuidado == "No Seguro": val_cuidado = 0.5
        else: val_cuidado = 0.0
        
        # Variable 5: Debilidad Social
        val_social = 1.0 if combo_social.get() == "Sí" else 0.0

        # 2. Calcular la fórmula matemática (z)
        z = intercepto + \
            (coef[0] * val_familia) + \
            (coef[1] * val_estres) + \
            (coef[2] * val_personal) + \
            (coef[3] * val_cuidado) + \
            (coef[4] * val_social)

        # 3. Función Sigmoide
        probabilidad = 1 / (1 + np.exp(-z))
        prob_porcentaje = probabilidad * 100

        # 4. Mostrar resultado en la pantalla
        lbl_resultado_num.config(text=f"Probabilidad: {prob_porcentaje:.2f}%")
        
        if probabilidad >= 0.5:
            lbl_conclusion.config(text="REQUIERE TRATAMIENTO", fg="red")
            lbl_icono.config(text="⚠️", fg="red")
        else:
            lbl_conclusion.config(text="Riesgo Bajo / Saludable", fg="green")
            lbl_icono.config(text="✅", fg="green")

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {e}")

# --- Configuración de la Ventana ---
ventana = tk.Tk()
ventana.title("Sistema Experto de Salud Mental")
ventana.geometry("400x550")
ventana.configure(bg="#f0f0f0") # Color de fondo gris claro

# Estilo de fuente
fuente_titulo = ("Arial", 16, "bold")
fuente_label = ("Arial", 10)

# Título
tk.Label(ventana, text="Diagnóstico IA Salud Mental", font=fuente_titulo, bg="#f0f0f0", pady=20).pack()

# --- Inputs (Preguntas) ---

# Marco para organizar mejor
frame_inputs = tk.Frame(ventana, bg="#ffffff", bd=2, relief="groove")
frame_inputs.pack(pady=10, padx=20, fill="x")

# Pregunta 1: Historial Familiar
tk.Label(frame_inputs, text="1. ¿Historial Familiar de Depresión?", bg="white", font=fuente_label).pack(anchor="w", padx=10, pady=(10,0))
combo_familia = ttk.Combobox(frame_inputs, values=["Sí", "No"], state="readonly")
combo_familia.current(1) # Seleccionar "No" por defecto
combo_familia.pack(fill="x", padx=10, pady=5)

# Pregunta 2: Estrés
tk.Label(frame_inputs, text="2. ¿Siente estrés creciente?", bg="white", font=fuente_label).pack(anchor="w", padx=10)
combo_estres = ttk.Combobox(frame_inputs, values=["Sí", "No"], state="readonly")
combo_estres.current(1)
combo_estres.pack(fill="x", padx=10, pady=5)

# Pregunta 3: Historial Personal
tk.Label(frame_inputs, text="3. ¿Historial Personal previo?", bg="white", font=fuente_label).pack(anchor="w", padx=10)
combo_personal = ttk.Combobox(frame_inputs, values=["Sí", "No", "Quizás"], state="readonly")
combo_personal.current(1)
combo_personal.pack(fill="x", padx=10, pady=5)

# Pregunta 4: Opciones de Cuidado
tk.Label(frame_inputs, text="4. ¿Conoce sus opciones de seguro médico?", bg="white", font=fuente_label).pack(anchor="w", padx=10)
combo_cuidado = ttk.Combobox(frame_inputs, values=["Sí", "No", "No Seguro"], state="readonly")
combo_cuidado.current(1)
combo_cuidado.pack(fill="x", padx=10, pady=5)

# Pregunta 5: Social
tk.Label(frame_inputs, text="5. ¿Siente aislamiento social?", bg="white", font=fuente_label).pack(anchor="w", padx=10)
combo_social = ttk.Combobox(frame_inputs, values=["Sí", "No"], state="readonly")
combo_social.current(1)
combo_social.pack(fill="x", padx=10, pady=(5, 10))

# --- Botón Calcular ---
btn_calcular = tk.Button(ventana, text="CALCULAR DIAGNÓSTICO", command=calcular_probabilidad, bg="#007bff", fg="white", font=("Arial", 11, "bold"), pady=10)
btn_calcular.pack(fill="x", padx=40, pady=20)

# --- Resultados ---
lbl_resultado_num = tk.Label(ventana, text="Probabilidad: --%", font=("Arial", 12), bg="#f0f0f0")
lbl_resultado_num.pack()

lbl_icono = tk.Label(ventana, text="", font=("Arial", 40), bg="#f0f0f0")
lbl_icono.pack()

lbl_conclusion = tk.Label(ventana, text="", font=("Arial", 14, "bold"), bg="#f0f0f0")
lbl_conclusion.pack(pady=10)

# Iniciar la ventana
ventana.mainloop()