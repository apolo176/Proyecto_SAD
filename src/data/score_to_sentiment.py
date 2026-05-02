#!/usr/bin/env python3
"""
Script de preprocesamiento y transformación inicial del dataset crudo.
Convierte puntuaciones numéricas en etiquetas categóricas de sentimiento y realiza el primer split de los datos.

CARACTERÍSTICAS CLAVE:
- Mapeo de la columna numérica 'score' (1-5) a etiquetas categóricas de 'sentiment' (NEGATIVO, NEUTRO, POSITIVO).
- División estratificada (Train/Test) garantizando el equilibrio representativo de las clases a partir de la configuración.
- Creación de los archivos base listos para ser consumidos por el resto de scripts de modelado o balanceo.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

from src.utils.funciones import load_config, load_data

def preparar_dataset():
    # 1. Cargar la configuración general
    config = load_config('config.json')

    # Extraemos la ruta del archivo original y el test_split
    # Como load_config aplana el JSON, podemos acceder directamente
    raw_file = config.get('raw_file', 'data/AppleMusic.csv')
    test_split = config.get('test_split', 0.20)

    plataforma_nombre = os.path.basename(raw_file).split('.')[0]
    print(f"Procesando datos de {plataforma_nombre}...")

    # 2. Cargar el archivo crudo
    try:
        df = pd.read_csv(raw_file)
    except FileNotFoundError as e:
        print(f"❌ Error: No se ha encontrado el archivo '{raw_file}'. {e}")
        return

    # 3. Mapeo de Score (1-5) a Sentimiento (Negativo, Neutro, Positivo)
    def mapear_sentimiento(score):
        try:
            s = int(score)
            if s <= 2: return 'NEGATIVO'
            if s == 3: return 'NEUTRO'
            return 'POSITIVO'
        except:
            return 'NEUTRO'

    if 'score' in df.columns:
        df['sentiment'] = df['score'].apply(mapear_sentimiento)
        print("✓ Transformación de score a sentimiento completada.")
    else:
        print("⚠️ Error: La columna 'score' no existe en el archivo.")
        return

    # 4. Preparar carpeta de salida
    os.makedirs('data', exist_ok=True)

    # 5. División: TRAIN y TEST controlada por config.json
    config = load_config('config.json')
    test_split = config.get('test', {}).get('test_split', 0.20)

    # Se mantiene el parámetro 'stratify' para asegurar una distribución equitativa de clases
    train, test = train_test_split(
        df,
        test_size=test_split,
        stratify=df['sentiment'],
        random_state=config['general']['random_state']
    )

    # 6. Guardar archivos resultantes
    train.to_csv('data/train.csv', index=False, encoding='utf-8')
    test.to_csv('data/test.csv', index=False, encoding='utf-8')

    print(f"✅ Proceso finalizado para Apple Music:")
    print(f"   - data/train.csv: {len(train)} muestras ({(1 - test_split) * 100:.0f}%)")
    print(f"   - data/test.csv: {len(test)} muestras ({test_split * 100:.0f}%)")


if __name__ == "__main__":
    preparar_dataset()