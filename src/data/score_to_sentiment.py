import pandas as pd
from sklearn.model_selection import train_test_split
import os

from src.utils.funciones import load_config, load_data

def preparar_dataset():
    print("Procesando datos de Apple Music...")

    # 1. Cargar el archivo de Apple Music
    try:
        df = pd.read_csv('data/AppleMusic.csv')
    except FileNotFoundError as e:
        print(f"Error: No se ha encontrado 'data/AppleMusic.csv'. {e}")
        return

    # 2. Mapeo de Score (1-5) a Sentimiento (Negativo, Neutro, Positivo)
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

    # 3. Preparar carpeta de salida
    os.makedirs('data', exist_ok=True)

    # 4. División: TRAIN (80%) y TEST (20%) para hacer pruebas nosotros
    # Se mantiene el parámetro 'stratify' para asegurar una distribución equitativa de clases

    config = load_config('config.json')

    train, test = train_test_split(
        df,
        test_size=0.20,
        stratify=df['sentiment'],
        random_state= config['general']['random_state']
    )

    # 5. Guardar archivos resultantes
    train.to_csv('data/train.csv', index=False, encoding='utf-8')
    test.to_csv('data/test.csv', index=False, encoding='utf-8')

    print(f"✅ Proceso finalizado para Apple Music:")
    print(f"   - data/train.csv: {len(train)} muestras")
    print(f"   - data/test.csv: {len(test)} muestras")


if __name__ == "__main__":
    preparar_dataset()