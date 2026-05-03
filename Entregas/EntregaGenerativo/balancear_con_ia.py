#!/usr/bin/env python3
"""
Script de balanceo de datos generativo mediante Inteligencia Artificial.
Realiza oversampling de la clase minoritaria generando reseñas sintéticas (paráfrasis) utilizando Ollama.

CARACTERÍSTICAS CLAVE:
- División Train/Dev ANTES del oversampling para garantizar la ausencia total de Data Leakage, y aplicar la generación para oversampling solamente para el Train.
- Generación de datos sintéticos que mantienen el sentimiento original pero con variaciones léxicas y estructurales.
- Cálculo automático de la brecha entre clases y aplicación de límites seguros de generación basados en config.json.
- Creación de un conjunto de entrenamiento balanceado por IA, exportando el conjunto Dev completamente intacto.
"""

import pandas as pd
import ollama
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils.funciones import load_config, load_data

PROMPT_PARAFRASIS = """Reescribe la siguiente opinión de una app de música (Apple Music/Spotify) manteniendo exactamente el MISMO sentimiento, pero usando diferentes palabras y estructuras.
No añadas introducciones, confirmaciones ni comillas. Genera SOLO la reseña. El idioma a usar tiene que ser el INGLÉS.

REGLA DE FORMATO IMPORTANTE: Debes generar la reseña como un único párrafo continuo. NO utilices saltos de línea, retornos de carro ni viñetas.

Original: '{texto}'
Reescritura:"""


def balancear_dataset():
    print("🤖 Iniciando Oversampling Generativo (Libre de Data Leakage)...")

    # 1. Cargar configuración y datos
    config = load_config('config.json')
    ruta_datos = config['general']['data']['train_dev']
    col_texto = config['general']['text_features'][0]
    col_target = config['general']['column']

    df = load_data(ruta_datos)

    # 2. SPLIT ANTES DEL OVERSAMPLING (Crucial para evitar Data Leakage)
    X_train, X_dev, y_train, y_dev = train_test_split(
        df[[col_texto]], df[col_target],
        test_size=config['preprocessing']['dev_size'],
        stratify=df[col_target],
        random_state=config['general']['random_state']
    )

    # Unimos temporalmente X_train e y_train para manipularlos
    df_train = pd.concat([X_train, y_train], axis=1)

    conteo = df_train[col_target].value_counts()
    clase_minoritaria = conteo.idxmin()
    print(f"📊 Clase minoritaria detectada en TRAIN: '{clase_minoritaria}' con {conteo.min()} muestras.")

    # 3. Cálculo automático del balanceo
    conteo = df_train[col_target].value_counts()
    n_max = conteo.max()  # Cantidad de la clase mayoritaria
    n_min = conteo.min()  # Cantidad de la clase minoritaria
    clase_minoritaria = conteo.idxmin()

    # Calculamos la diferencia real
    diferencia = n_max - n_min

    # --- NUEVA LÓGICA: LÍMITE DE GENERACIÓN ---
    # Leemos el límite desde config.json (por defecto 200 si no existe)
    limite_ia = config['generative'].get('n_generations', 200)

    # El objetivo será la diferencia, PERO nunca superando el límite de la IA
    objetivo_generaciones = min(diferencia, limite_ia)

    print(f"📊 Clase mayoritaria: {n_max} muestras.")
    print(f"📊 Clase minoritaria: {n_min} muestras.")
    print(f"⚠️ Diferencia total: {diferencia} reseñas.")
    print(f"🧠 Objetivo IA: Generar {objetivo_generaciones} reseñas de tipo '{clase_minoritaria}' (Límite aplicado).")

    # Obtenemos los textos base (clase minoritaria) para que la IA los use de referencia
    textos_referencia = df_train[df_train[col_target] == clase_minoritaria][col_texto].tolist()

    nuevas_filas = []

    # Usamos un bucle que se repita exactamente 'objetivo_generaciones' veces
    for i in tqdm(range(objetivo_generaciones), desc="Generando balanceo"):
        # Elegimos un texto de referencia al azar de los que tenemos
        import random
        texto_original = random.choice(textos_referencia)

        prompt_final = PROMPT_PARAFRASIS.format(texto=texto_original)

        respuesta = ollama.chat(
            model="llama3",
            messages=[{'role': 'user', 'content': prompt_final}],
            options={'temperature': 0.8, 'num_predict': 100}
        )
        nueva_review = respuesta['message']['content']
        # --- LIMPIEZA ANTI-CSV-BREAKING ---
        # 1. Reemplazar saltos de línea reales (\n) y retornos de carro (\r) por un espacio simple
        nueva_review = nueva_review.replace('\n', ' ').replace('\r', ' ')
        # 2. Limpiar espacios en blanco sobrantes a los lados
        nueva_review = nueva_review.strip()
        # ---------------------------------

        nuevas_filas.append({col_texto: nueva_review, col_target: clase_minoritaria})

    # 4. Guardar el entregable de paráfrasis
    df_nuevas = pd.DataFrame(nuevas_filas)
    df_nuevas.to_csv('resultados/parafrases_generadas_ollama.csv', index=False, encoding='utf-8')
    print("\n📁 Archivo de entregable generado: 'resultados/parafrases_generadas_ollama.csv'")

    # 5. Generar un nuevo Train balanceado (y guardar Dev intacto por si tus compañeros lo necesitan)
    df_train_balanceado = pd.concat([df_train, df_nuevas], ignore_index=True)
    df_train_balanceado.to_csv('data/train_SOLO_balanceado_ia.csv', index=False, encoding='utf-8')

    # Guardamos el Dev correspondiente para que si validan contra este Train, usen el Dev correcto
    df_dev = pd.concat([X_dev, y_dev], axis=1)
    df_dev.to_csv('data/dev_intacto.csv', index=False, encoding='utf-8')

    print("🎉 ¡Datasets guardados sin Data Leakage!")
    print(f"   - Nuevo Train balanceado: data/train_SOLO_balanceado_ia.csv")
    print(f"   - Validation (Dev) intacto: data/dev_intacto.csv")
    print("\nNuevo conteo de clases en TRAIN:")
    print(df_train_balanceado[col_target].value_counts())


if __name__ == "__main__":
    balancear_dataset()