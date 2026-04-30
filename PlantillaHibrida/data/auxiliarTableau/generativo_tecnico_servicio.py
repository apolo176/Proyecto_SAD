import argparse
import pandas as pd
import ollama
from tqdm import tqdm
import os

# ==========================================
# 1. DEFINICIÓN DE LA PLANTILLA DE PROMPT
# ==========================================

PROMPT_CLASIFICACION_REVIEWS = """Act as an expert product and user feedback analyst.
Your task is to classify the following application review strictly into one of these three categories:

1 (Technical): Software/hardware issues, performance, bugs, or technical features (e.g., audio quality, RAM usage, crashes, UI glitches).
2 (Service): Business-related matters, content, pricing, subscriptions, advertisements, customer support, or catalog availability.
0 (None): Generic comments, emotions without specific context, or vague complaints lacking details.

Rule: Return ONLY the number (0, 1, or 2). Do not provide explanations or add any other characters.

Reference examples:
Review: 'It has lossless audio.'
Category: 1

Review: 'It crashes on android.'
Category: 1

Review: 'It consumes a lot of ram.'
Category: 1

Review: 'The UI freezes when I scroll down.'
Category: 1

Review: 'Battery drains extremely fast while using this app.'
Category: 1

Review: 'It has all my favorite artists!'
Category: 2

Review: 'The subscription is too expensive.'
Category: 2

Review: 'It has too many ads.'
Category: 2

Review: 'Customer support never replied to my refund request.'
Category: 2

Review: 'They removed the offline listening feature for free users.'
Category: 2

Review: 'Great! I love it.'
Category: 0

Review: 'It doesn't work.'
Category: 0

Review: 'Absolute trash.'
Category: 0

Review: 'Five stars, totally recommended.'
Category: 0

Review: '{texto}'
Category:"""

# ==========================================
# 2. FUNCIÓN DE INFERENCIA MEDIANTE OLLAMA
# ==========================================

def interactuar_ollama(prompt, modelo="llama3", temperatura=0.0, limite_tokens=5):
    """
    Transmite el prompt al modelo lingüístico a través de Ollama y extrae 
    exclusivamente el primer carácter numérico válido para evitar ruido en la inferencia.
    """
    try:
        response = ollama.chat(model=modelo, messages=[
            {'role': 'user', 'content': prompt}
        ], options={
            'temperature': temperatura,
            'num_predict': limite_tokens
        })
        
        respuesta_cruda = response['message']['content'].strip()
        
        for char in respuesta_cruda:
            if char in ['0', '1', '2']:
                return int(char)
        
        return -1 
        
    except Exception as e:
        print(f"Error de comunicación con el modelo de inferencia: {e}")
        return -1

# ==========================================
# 3. PROCESAMIENTO DEL CONJUNTO DE DATOS
# ==========================================

def clasificar_dataset(ruta_entrada, ruta_salida, columna_texto, modelo_base, rango_filas):
    """
    Procesa una ventana específica de registros sin descartar el resto del conjunto de datos.
    Permite la actualización incremental de la columna 'tipo_review'.
    """
    print(f"Iniciando el proceso de clasificación empleando el modelo '{modelo_base}'...")

    try:
        df = pd.read_csv(ruta_entrada)
    except Exception as e:
        print(f"Error crítico al leer el archivo de entrada: {e}")
        return

    if columna_texto not in df.columns:
        print(f"Error crítico: La columna '{columna_texto}' no existe en el archivo.")
        return

    # Inicialización de la columna de resultados si no está presente en el archivo de origen
    if 'tipo_review' not in df.columns:
        df['tipo_review'] = -1  # Valor por defecto para registros no procesados

    # Determinación de los índices de la ventana de acción
    if rango_filas:
        try:
            inicio, fin = map(int, rango_filas.split('-'))
            # Validación de límites para evitar desbordamiento de índice
            fin = min(fin, len(df))
            indices_objetivo = range(inicio, fin)
            print(f"Ventana de acción definida: índices {inicio} a {fin-1} ({len(indices_objetivo)} registros).")
        except ValueError:
            print("Error: Formato de rango inválido. Use 'inicio-fin'.")
            return
    else:
        indices_objetivo = df.index
        print(f"Procesando la totalidad de los registros ({len(df)} filas).")

    # Bucle de procesamiento selectivo
    for i in tqdm(indices_objetivo, desc="Clasificando"):
        texto = str(df.at[i, columna_texto])
        prompt_final = PROMPT_CLASIFICACION_REVIEWS.format(texto=texto)
        categoria = interactuar_ollama(prompt_final, modelo=modelo_base)
        df.at[i, 'tipo_review'] = categoria

    # Persistencia del conjunto de datos completo (incluyendo filas no procesadas)
    try:
        df.to_csv(ruta_salida, index=False, encoding='utf-8')
        print(f"\nProceso finalizado. El archivo completo ha sido guardado en: {ruta_salida}")
        print("Resumen de la columna 'tipo_review' (incluye registros previos y pendientes):")
        print(df['tipo_review'].value_counts())
    except Exception as e:
        print(f"Error crítico al guardar el archivo de salida: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clasificación incremental de reviews mediante Ollama."
    )
    
    parser.add_argument("-i", "--input", required=True, help="Archivo CSV de entrada.")
    parser.add_argument("-o", "--output", required=True, help="Archivo CSV de salida.")
    parser.add_argument("-c", "--column", default="review", help="Nombre de la columna de texto.")
    parser.add_argument("-m", "--model", default="llama3", help="Modelo de Ollama.")
    parser.add_argument("-r", "--range", default=None, help="Rango base 0 (ej. '0-100').")
    
    args = parser.parse_args()
    
    clasificar_dataset(args.input, args.output, args.column, args.model, args.range)
