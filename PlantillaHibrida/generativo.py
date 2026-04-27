import pandas as pd
import ollama
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

from funciones import load_config, load_data

# ==========================================
# 1. PLANTILLAS DE PROMPTS (Evolución y Pruebas)
# ==========================================

# --- EXPERIMENTOS ZERO-SHOT ---

PROMPT_ZERO_SHOT_MALO = """Dime el sentimiento de este comentario de una app.
Comentario: '{texto}'
Respuesta:"""

PROMPT_ZERO_SHOT_NORMAL = """Clasifica el siguiente texto en POSITIVO, NEGATIVO o NEUTRO.
Intenta no decir nada más, solo la palabra.
Texto: '{texto}'
Sentimiento:"""

PROMPT_ZERO_SHOT_BUENO = """Actúa como un experto en análisis de sentimientos.
Clasifica el siguiente texto de una app estrictamente en una de estas tres categorías:
A (POSITIVO)
B (NEUTRO)
C (NEGATIVO)
Regla: Devuelve ÚNICAMENTE la letra correspondiente (A, B o C). No des explicaciones ni uses puntuación.

Texto: '{texto}'
Sentimiento:"""

# --- EXPERIMENTOS ONE-SHOT ---

PROMPT_ONE_SHOT_MALO = """Clasifica el texto en Positivo, Negativo o Neutro.
Ejemplo de cómo no funciona bien: "Esta app es un desastre" -> Creo que es negativo.
Ahora hazlo tú con este: '{texto}'
Sentimiento:"""

PROMPT_ONE_SHOT_BUENO = """Actúa como un experto en análisis de sentimientos.
Clasifica el texto estrictamente en una de estas tres categorías:
A (POSITIVO)
B (NEUTRO)
C (NEGATIVO)
Regla: Devuelve ÚNICAMENTE la letra correspondiente (A, B o C). No des explicaciones ni uses puntuación.

Texto: 'La app no me deja hacer match, se cuelga todo el rato.'
Sentimiento: C

Texto: '{texto}'
Sentimiento:"""

# --- EXPERIMENTOS FEW-SHOT ---

PROMPT_FEW_SHOT_ADOLESCENTE = """Actúa como un adolescente en redes sociales. 
Dime si estos comentarios sobre una app molan (POSITIVO), dan igual (NEUTRO) o son un rollo (NEGATIVO).
'se cuelga' -> NEGATIVO
'es la caña' -> POSITIVO
'es azul' -> NEUTRO
'{texto}' ->"""

PROMPT_FEW_SHOT_BUENO = """Actúa como un experto en análisis de sentimientos.
Clasifica el texto estrictamente en una de estas tres categorías:
A (POSITIVO)
B (NEUTRO)
C (NEGATIVO)
Regla: Devuelve ÚNICAMENTE la letra correspondiente (A, B o C). No des explicaciones ni uses puntuación.

Texto: 'La app no me deja hacer match, se cuelga todo el rato.'
Sentimiento: C

Texto: 'Conocí a mi pareja aquí, la selección musical es increíble.'
Sentimiento: A

Texto: 'La interfaz es de color azul y tiene un menú lateral.'
Sentimiento: B

Texto: '{texto}'
Sentimiento:"""


# ==========================================
# 2. FUNCIÓN DE COMUNICACIÓN CON OLLAMA
# ==========================================

def interactuar_ollama(prompt, modelo="llama3:8b", temperatura=0.0, limite_tokens=15):
    """
    Envía el prompt a Ollama.
    """
    try:
        response = ollama.chat(model=modelo, messages=[
            {'role': 'user', 'content': prompt}
        ], options={
            'temperature': temperatura,
            'num_predict': limite_tokens
        })
        return response['message']['content'].strip()
    except Exception as e:
        return f"ERROR: {e}"


# ==========================================
# 3. GENERACIÓN DEL ENTREGABLE CSV
# ==========================================

def generar_entregable():
    print("🚀 Iniciando experimentos con Ollama integrados con Plantilla Híbrida...")

    # 1. Cargar Configuración y Datos
    config = load_config('config.json')
    ruta_datos = config['general']['data']['train_dev']
    columna_texto = config['general']['text_features'][0]
    columna_target = config['general']['column']

    df = load_data(ruta_datos)

    # 2. Replicar el split exacto que hace train.py para no tener Data Leakage
    test_size = config['preprocessing']['test_size']
    random_state = config['general']['random_state']

    X_train, X_dev, y_train, y_dev = train_test_split(
        df[[columna_texto]], df[columna_target],
        test_size=test_size,
        stratify=df[columna_target],
        random_state=random_state
    )

    # 3. Tomar una muestra de DEV para evaluar (ej. 50 registros para no tardar horas)
    muestra_dev = X_dev.head(50).copy()
    y_verdadero = y_dev.head(50).tolist()
    textos_prueba = muestra_dev[columna_texto].tolist()

    mapa_sentimientos = {'A': 'positivo', 'B': 'neutro',
                         'C': 'negativo'}  # Ajustado a minúsculas si tu dataset está así

    experimentos = [
        {"id": "Zero-shot (Malo)", "plantilla": PROMPT_ZERO_SHOT_MALO, "tokens": 20, "mapear": False},
        {"id": "Zero-shot (Normal)", "plantilla": PROMPT_ZERO_SHOT_NORMAL, "tokens": 10, "mapear": False},
        {"id": "Zero-shot (Bueno)", "plantilla": PROMPT_ZERO_SHOT_BUENO, "tokens": 1, "mapear": True},
        {"id": "One-shot (Malo)", "plantilla": PROMPT_ONE_SHOT_MALO, "tokens": 20, "mapear": False},
        {"id": "One-shot (Bueno)", "plantilla": PROMPT_ONE_SHOT_BUENO, "tokens": 1, "mapear": True},
        {"id": "Few-shot (Rol Adolescente)", "plantilla": PROMPT_FEW_SHOT_ADOLESCENTE, "tokens": 15, "mapear": False},
        {"id": "Few-shot (Bueno)", "plantilla": PROMPT_FEW_SHOT_BUENO, "tokens": 1, "mapear": True}
    ]

    modelo_base = "llama3"  # Ajustado a tu modelo local
    resultados = []
    predicciones_mejor_modelo = []

    for exp in experimentos:
        print(f"\nProbando: {exp['id']}")
        es_el_mejor = exp['id'] == "Few-shot (Bueno)"
        etiqueta_modelo = f"[MEJOR RESULTADO] {modelo_base} ({exp['id']})" if es_el_mejor else f"{modelo_base} ({exp['id']})"

        for texto in tqdm(textos_prueba, desc="Procesando"):
            prompt_final = exp['plantilla'].format(texto=texto)
            salida_raw = interactuar_ollama(prompt_final, modelo_base, 0.0, exp['tokens'])

            if exp['mapear']:
                letra_limpia = salida_raw.upper().strip()
                salida_final = mapa_sentimientos.get(letra_limpia, f"ERROR_DE_MAPEO (Raw: {salida_raw})")
                if es_el_mejor:
                    predicciones_mejor_modelo.append(salida_final)
            else:
                salida_final = salida_raw.replace("\n", " ")

            resultados.append({
                "Modelo y Tamaño": etiqueta_modelo,
                "Prompt Empleado": prompt_final,
                "Entrada": texto,
                "Salida": salida_final
            })

    # Calcular métrica para el mejor modelo y mostrarla
    if len(predicciones_mejor_modelo) == len(y_verdadero):
        acc = accuracy_score(y_verdadero, predicciones_mejor_modelo)
        print(f"\n📊 Accuracy del Mejor Modelo Generativo en DEV (50 muestras): {acc:.4f}")

    # Guardar CSV (como ya tenías)
    df_res = pd.DataFrame(resultados)
    df_mejor = df_res[df_res['Modelo y Tamaño'].str.contains(r'\[MEJOR RESULTADO\]', regex=True)]
    df_resto = df_res[~df_res['Modelo y Tamaño'].str.contains(r'\[MEJOR RESULTADO\]', regex=True)]
    df_final = pd.concat([df_mejor, df_resto], ignore_index=True)

    columnas_finales = ["Modelo y Tamaño", "Prompt Empleado", "Entrada", "Salida"]
    df_final[columnas_finales].to_csv("resultados_generativos.csv", index=False, encoding='utf-8')
    print("\n✅ Archivo 'resultados_generativos.csv' generado con éxito.")


if __name__ == "__main__":
    generar_entregable()