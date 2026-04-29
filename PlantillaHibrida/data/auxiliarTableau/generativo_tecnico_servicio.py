import argparse
import pandas as pd
import ollama
from tqdm import tqdm

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
        
        # Depuración de la cadena de salida para aislar la clasificación categórica
        respuesta_cruda = response['message']['content'].strip()
        
        # Identificación del primer dígito válido en la secuencia
        for char in respuesta_cruda:
            if char in ['0', '1', '2']:
                return int(char)
        
        # Retorna -1 si el modelo genera una alucinación u omite la clase requerida
        return -1 
        
    except Exception as e:
        print(f"Error de comunicación con el modelo de inferencia: {e}")
        return -1

# ==========================================
# 3. PROCESAMIENTO DEL CONJUNTO DE DATOS
# ==========================================

def clasificar_dataset(ruta_entrada, ruta_salida, columna_texto, modelo_base):
    """
    Lee el conjunto de datos, itera sobre los registros aplicando el modelo de 
    clasificación especificado y almacena los resultados en un nuevo archivo en formato CSV.
    """
    print(f"Iniciando el proceso de clasificación de texto empleando el modelo '{modelo_base}'...")

    # Carga de los datos en memoria
    try:
        df = pd.read_csv(ruta_entrada)
    except FileNotFoundError:
        print(f"Error crítico: No se ha localizado el archivo '{ruta_entrada}'. Verifique la ruta proporcionada.")
        return
    except Exception as e:
        print(f"Error crítico al leer el archivo de entrada: {e}")
        return

    if columna_texto not in df.columns:
        print(f"Error crítico: La columna '{columna_texto}' no existe en el conjunto de datos provisto.")
        return

    clasificaciones = []

    # Iteración y clasificación secuencial de cada registro
    for texto in tqdm(df[columna_texto].astype(str), desc="Procesando registros"):
        prompt_final = PROMPT_CLASIFICACION_REVIEWS.format(texto=texto)
        categoria = interactuar_ollama(prompt_final, modelo=modelo_base)
        clasificaciones.append(categoria)

    # Integración de los resultados de inferencia en el marco de datos original
    df['tipo_review'] = clasificaciones

    # Evaluación de la integridad estructural de los resultados
    errores = df[df['tipo_review'] == -1]
    if len(errores) > 0:
        print(f"Advertencia: Se han detectado {len(errores)} registros que no pudieron ser clasificados adecuadamente por el modelo.")

    # Persistencia del conjunto de datos resultante
    try:
        df.to_csv(ruta_salida, index=False, encoding='utf-8')
        print("Proceso de clasificación finalizado con éxito.")
        print("\nDistribución de clases resultantes:")
        print(df['tipo_review'].value_counts())
        print(f"\nLos resultados han sido almacenados correctamente en: {ruta_salida}")
    except Exception as e:
        print(f"Error crítico al intentar guardar el archivo de salida: {e}")

if __name__ == "__main__":
    # Configuración del analizador de argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Herramienta de clasificación de comentarios de usuarios utilizando modelos de lenguaje locales (Ollama)."
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Ruta absoluta o relativa al archivo CSV de entrada que contiene los textos a analizar."
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Ruta absoluta o relativa al archivo CSV de salida para almacenar los resultados del análisis."
    )
    
    parser.add_argument(
        "-c", "--column", 
        default="review", 
        help="Nombre de la variable/columna que contiene el texto en el CSV (valor por defecto: 'review_text')."
    )

    parser.add_argument(
        "-m", "--model", 
        default="llama3", 
        help="Nombre del modelo de lenguaje de Ollama a utilizar para la inferencia (valor por defecto: 'llama3')."
    )
    
    args = parser.parse_args()
    
    # Ejecución de la rutina principal con los parámetros provistos
    clasificar_dataset(args.input, args.output, args.column, args.model)
