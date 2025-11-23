import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Configuración de logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MentalHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción de Salud Mental - IA")
        # Hacemos la ventana más ancha para que quepan las dos columnas
        self.root.geometry("1000x650")
        
        # --- 1. CARGA DE ARTEFACTOS ---
        try:
            self.model = tf.keras.models.load_model('modelo_salud_mental.keras')
            self.scaler = joblib.load('scaler_salud_mental.pkl')
            self.model_columns = joblib.load('columnas_entrenamiento.pkl')
            print("Sistema cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error Crítico", f"Faltan archivos del modelo.\nVerifica que tengas el .keras y los .pkl en la carpeta.\n\nDetalle: {e}")
            self.root.destroy()
            return

        # --- 2. DICCIONARIOS DE TRADUCCIÓN ---
        self.traductor_interno = {
            "Sí": "Yes", "No": "No", "Tal vez": "Maybe", "No estoy seguro": "Not sure",
            "Masculino": "Male", "Femenino": "Female", "No Binario": "Non-binary",
            "Corporativo": "Corporate", "Estudiante": "Student", "Negocios/Empresario": "Business", 
            "Ama de casa": "Housewife", "Otros": "Others",
            "1-14 días": "1-14 days", "15-30 días": "15-30 days", "31-60 días": "31-60 days", 
            "Más de 2 meses": "More than 2 months", "Salgo todos los días": "Go out Every day",
            "Bajo": "Low", "Medio": "Medium", "Alto": "High"
        }

        # --- 3. INTERFAZ GRÁFICA (DOS COLUMNAS) ---
        
        # Título
        lbl_title = tk.Label(root, text="Diagnóstico de Salud Mental (IA)", font=("Segoe UI", 18, "bold"), fg="#333")
        lbl_title.pack(pady=15)

        # Contenedor Principal
        main_container = tk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        # --- CREACIÓN DE PANELES (IZQUIERDA Y DERECHA) ---
        
        # Panel Izquierdo: Datos Personales
        self.frame_izq = tk.LabelFrame(main_container, text=" Datos Generales ", font=("Arial", 11, "bold"), fg="#0056b3")
        self.frame_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Panel Derecho: Historial y Psicología (Para llenar el espacio vacío)
        self.frame_der = tk.LabelFrame(main_container, text=" Historial y Psicología ", font=("Arial", 11, "bold"), fg="#0056b3")
        self.frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.entries = {}
        opts_si_no = ["Sí", "No", "Tal vez", "No estoy seguro"]

        # --- DISTRIBUCIÓN DE CAMPOS ---

        # COLUMNA IZQUIERDA (7 campos)
        self.crear_campo(self.frame_izq, "País de Residencia", "Country", "text")
        self.crear_campo(self.frame_izq, "Género", "Gender", "combo", ["Masculino", "Femenino", "No Binario"])
        self.crear_campo(self.frame_izq, "Ocupación", "Occupation", "combo", ["Estudiante", "Corporativo", "Negocios/Empresario", "Ama de casa", "Otros"])
        self.crear_campo(self.frame_izq, "¿Trabaja por cuenta propia?", "self_employed", "combo", opts_si_no)
        self.crear_campo(self.frame_izq, "¿Días sin salir de casa?", "Days_Indoors", "combo", ["1-14 días", "15-30 días", "31-60 días", "Más de 2 meses", "Salgo todos los días"])
        self.crear_campo(self.frame_izq, "¿Cambios de humor recientes?", "Mood_Swings", "combo", ["Bajo", "Medio", "Alto"])
        self.crear_campo(self.frame_izq, "¿Tiene usted EPS/Seguro?", "care_options", "combo", ["No estoy seguro", "No", "Sí"])

        # COLUMNA DERECHA (9 campos - Llenando el espacio rojo)
        self.crear_campo(self.frame_der, "¿Historial familiar de depresión?", "family_history", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Tratamiento previo?", "treatment", "combo", ["Sí", "No"])
        self.crear_campo(self.frame_der, "¿Siente estrés creciente?", "Growing_Stress", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Cambios en sueño/comida?", "Changes_Habits", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Historial personal mental?", "Mental_Health_History", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Le cuesta manejar problemas?", "Coping_Struggles", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Interés laboral?", "Work_Interest", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Debilidad social (aislamiento)?", "Social_Weakness", "combo", opts_si_no)
        self.crear_campo(self.frame_der, "¿Entrevista previa?", "mental_health_interview", "combo", ["No", "Tal vez", "Sí"])

        # Botón (Abajo del todo)
        btn_calc = tk.Button(root, text="CALCULAR DIAGNÓSTICO", bg="#007bff", fg="white", font=("Segoe UI", 12, "bold"), height=2, command=self.procesar_datos)
        btn_calc.pack(pady=20, fill=tk.X, padx=50, side=tk.BOTTOM)

    # --- MÉTODO MODIFICADO PARA ACEPTAR "parent" ---
    def crear_campo(self, parent, texto_etiqueta, llave_interna, tipo, opciones=None):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5) # Padding vertical para que no se vean amontonados
        
        lbl = tk.Label(frame, text=texto_etiqueta, width=28, anchor="w")
        lbl.pack(side=tk.LEFT)
        
        if tipo == "text":
            wid = tk.Entry(frame)
        elif tipo == "combo":
            wid = ttk.Combobox(frame, values=opciones, state="readonly")
            wid.current(0)
            
        wid.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        self.entries[llave_interna] = wid

    def procesar_datos(self):
        # 1. Obtener datos y Traducir
        datos_traducidos = {}
        
        for key, widget in self.entries.items():
            valor_usuario = widget.get()
            valor_final = self.traductor_interno.get(valor_usuario, valor_usuario)
            datos_traducidos[key] = [valor_final]

        df_input = pd.DataFrame(datos_traducidos)

        # 2. Preprocesamiento
        try:
            # País
            pais = str(df_input['Country'].iloc[0]).lower()
            if pais in ['united states', 'estados unidos', 'usa', 'eeuu']:
                df_input['Country'] = 1
            else:
                df_input['Country'] = 0

            # Mapeo Yes/No
            cols_yes_no = ['self_employed', 'family_history', 'Growing_Stress', 
                           'Changes_Habits', 'Mental_Health_History', 'Coping_Struggles', 
                           'Work_Interest', 'Social_Weakness']
            
            map_modelo = {'Yes': 1, 'No': 0, 'Maybe': 0.5, 'Not sure': 0.5}
            
            for col in cols_yes_no:
                if col in df_input.columns:
                    df_input[col] = df_input[col].map(map_modelo).fillna(0)

            # One-Hot Encoding
            cols_encode = ['Gender', 'Occupation', 'Days_Indoors', 'Mood_Swings', 
                           'mental_health_interview', 'care_options']
            
            df_input = pd.get_dummies(df_input, columns=cols_encode, drop_first=True)

            # Reindexar (CRÍTICO)
            df_input = df_input.reindex(columns=self.model_columns, fill_value=0)
            
            # Escalar
            X_scal = self.scaler.transform(df_input.values)

            # 3. Predicción
            prob = self.model.predict(X_scal)[0][0]
            
            # 4. Resultado
            resultado = "REQUIERE ATENCIÓN PROFESIONAL" if prob > 0.5 else "NO REQUIERE TRATAMIENTO URGENTE"
            color_icono = "warning" if prob > 0.5 else "info"
            
            msg = f"Análisis completado:\n\nDiagnóstico: {resultado}\nProbabilidad calculada: {prob:.1%}"
            messagebox.showinfo("Resultados del Modelo", msg, icon=color_icono)

        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Ocurrió un error interno:\n{e}")
            print(e)

if __name__ == "__main__":
    root = tk.Tk()
    app = MentalHealthApp(root)
    root.mainloop()