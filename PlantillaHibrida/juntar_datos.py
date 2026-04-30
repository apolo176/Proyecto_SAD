import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ----------------------------------------------------------
# IMPORTANTE (LIVIU, 17:35 - 28/04/26): Creo que no hace falta juntar los dataset de las 2 aplicaciones, si no que hay que hacerlo por separado la clasificación,
# primero de Apple Music y luego de Spotify
# ----------------------------------------------------------

def preparar_dataset():
    print("Files processing...")

    # 1. Cargar los archivos originales
    # Asumimos que están en la carpeta data/
    try:
        apple = pd.read_csv('data/AppleMusic.csv')
        spotify = pd.read_csv('data/Spotify.csv')
    except FileNotFoundError as e:
        print(f"Error: No se han encontrado los archivos en la carpeta data/. {e}")
        return

    # 2. Añadir identificador de App (Fundamental para Tableau)
    apple['App'] = 'Apple Music'
    spotify['App'] = 'Spotify'

    # 3. Unir los datasets
    df_total = pd.concat([apple, spotify], ignore_index=True)
    print(f"✓ Datos combinados: {len(df_total)} reseñas totales.")

    # 4. Mapeo de Score (1-5) a Sentimiento (Negativo, Neutro, Positivo)
    # Según el Enunciado.pdf, el objetivo es predecir estas 3 categorías
    def mapear_sentimiento(score):
        try:
            s = int(score)
            if s <= 2: return 'NEGATIVO'
            if s == 3: return 'NEUTRO'
            return 'POSITIVO'
        except:
            return 'NEUTRO'

    columna_origen = 'score'
    if columna_origen in df_total.columns:
        df_total['sentiment'] = df_total[columna_origen].apply(mapear_sentimiento)
        print("✓ Conversión de score a 3 clases completada.")
    else:
        print("⚠️ Advertencia: No se encontró la columna de puntuación original.")

    # 5. Guardar el archivo MAESTRO para Tableau (con todas las columnas y la App)
    os.makedirs('data', exist_ok=True)
    df_total.to_csv('data/dataset_completo_tableau.csv', index=False, encoding='utf-8')
    print("💾 Archivo para Tableau guardado en data/dataset_completo_tableau.csv")

    # 6. División Final: TRAIN (80%) y TEST (20%)
    # El test.csv es el que se usará al final en test.py
    train, test = train_test_split(
        df_total,
        test_size=0.20,
        stratify=df_total['sentiment'],
        random_state=42
    )

    train.to_csv('data/train.csv', index=False, encoding='utf-8')
    test.to_csv('data/test.csv', index=False, encoding='utf-8')

    print(f"✅ Proceso finalizado:")
    print(f"   - data/train.csv: {len(train)} muestras")
    print(f"   - data/test.csv: {len(test)} muestras")


if __name__ == "__main__":
    preparar_dataset()