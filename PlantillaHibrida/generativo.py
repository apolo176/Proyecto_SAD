import pandas as pd
import ollama
from tqdm import tqdm
import time

# ==========================================
# 1. PLANTILLAS DE PROMPTS MEJORADAS
# ==========================================

# ZERO-SHOT: Asignamos rol, idioma 100% español e instrucciones de prohibición claras.
PROMPT_ZERO_SHOT = """Actúa como un experto en análisis de sentimientos de usuarios.
Clasifica el siguiente texto de una app de citas/música estrictamente en una de estas tres categorías: POSITIVO, NEGATIVO o NEUTRO.
Regla: No des explicaciones, no uses puntuación. Devuelve ÚNICAMENTE la palabra de la categoría.

Texto: '{texto}'
Sentimiento:"""

# ONE-SHOT: Le damos exactamente UN ejemplo para que entienda el patrón de respuesta.
PROMPT_ONE_SHOT = """Actúa como un experto en análisis de sentimientos de usuarios.
Clasifica el texto estrictamente en una de estas tres categorías: POSITIVO, NEGATIVO o NEUTRO.
Regla: No des explicaciones, no uses puntuación. Devuelve ÚNICAMENTE la palabra de la categoría.

Texto: 'La app no me deja hacer match, se cuelga todo el rato.'
Sentimiento: NEGATIVO

Texto: '{texto}'
Sentimiento:"""

# FEW-SHOT: Le damos un ejemplo de CADA clase (el mejor formato posible).
PROMPT_FEW_SHOT = """Actúa como un experto en análisis de sentimientos de usuarios.
Clasifica el texto estrictamente en una de estas tres categorías: POSITIVO, NEGATIVO o NEUTRO.
Regla: No des explicaciones, no uses puntuación. Devuelve ÚNICAMENTE la palabra de la categoría.

Texto: 'La app no me deja hacer match, se cuelga todo el rato.'
Sentimiento: NEGATIVO

Texto: 'Conocí a mi pareja aquí, la selección musical es increíble.'
Sentimiento: POSITIVO

Texto: 'La interfaz es de color azul y tiene un menú lateral.'
Sentimiento: NEUTRO

Texto: '{texto}'
Sentimiento:"""

# PARÁFRASIS (OVERSAMPLING): Estructura sólida para inventar datos sin cambiar la clase original.
PROMPT_PARAFRASIS = """Reescribe la siguiente opinión de un usuario sobre una app de citas y música.
Reglas:
1. Mantén exactamente el MISMO sentimiento (positivo, negativo o neutro) que el original.
2. Usa palabras y estructuras gramaticales completamente diferentes (sé creativo).
3. No añadas introducciones, confirmaciones ni comillas. Genera SOLO el texto de la reseña.

Original: '{texto}'
Reescritura:"""


# ==========================================
# 2. FUNCIÓN DE COMUNICACIÓN CON OLLAMA
# ==========================================

def interactuar_ollama(prompt, modelo="llama3:8b-text-q2_K", temperatura=0.0):
    """
    Envía el prompt a Ollama.
    Usamos temperatura=0.0 para clasificación (determinístico, como pide la profe)
    y temperatura > 0.8 para paráfrasis (para que sea creativo).
    """
    try:
        # En la API de Python, pasamos los "parameters" (del PDF) en el objeto 'options'
        response = ollama.chat(model=modelo, messages=[
            {'role': 'user', 'content': prompt}
        ], options={
            'temperature': temperatura,
            'num_predict': 150  # Límite de tokens para que no "hable de más" (mencionado en el PDF)
        })
        return response['message']['content'].strip()
    except Exception as e:
        return f"ERROR: {e}"


# ==========================================
# 3. GENERACIÓN DEL ENTREGABLE CSV
# ==========================================

def generar_entregable():
    print("🚀 Iniciando experimentos con Ollama...")

    # Textos de prueba (puedes cambiarlos por algunos reales de vuestro train.csv)
    textos_prueba = [
        "No carga la música en los perfiles, es desesperante.",
        "Me gusta el diseño, es fácil de usar.",
        "Simplemente es una app de citas."
    ]

    experimentos = [
        {"modelo": "llama3", "tipo": "Clasificación (Zero-shot)", "plantilla": PROMPT_ZERO_SHOT, "temp": 0.0},
        {"modelo": "llama3", "tipo": "Clasificación (One-shot)", "plantilla": PROMPT_ONE_SHOT, "temp": 0.0},
        {"modelo": "llama3", "tipo": "Clasificación (Few-shot)", "plantilla": PROMPT_FEW_SHOT, "temp": 0.0},
        {"modelo": "llama3", "tipo": "Generación/Paráfrasis", "plantilla": PROMPT_PARAFRASIS, "temp": 0.8}
    ]

    resultados = []

    for exp in experimentos:
        print(f"\nProbando: {exp['tipo']} con temp={exp['temp']}")
        for texto in tqdm(textos_prueba, desc="Procesando"):
            inicio = time.time()

            # Insertar el texto en la plantilla
            prompt_final = exp['plantilla'].format(texto=texto)

            # Llamada a la IA
            salida = interactuar_ollama(prompt_final, exp['modelo'], exp['temp'])

            tiempo = round(time.time() - inicio, 2)

            resultados.append({
                "Modelo y Tamaño": exp['modelo'],
                "Prompt Empleado": prompt_final,
                "Entrada": texto,
                "Salida": salida,
                "Tiempo (s)": tiempo
            })

    # Guardar el CSV que pide la profesora
    df = pd.DataFrame(resultados)
    # Según la rúbrica: "colocar al comienzo y resaltar el modelo y el prompt que ha funcionado mejor"
    # Luego tú en Excel puedes poner "[MEJOR]" en la fila que mejor haya rendido.
    df.to_csv("resultados_generativos.csv", index=False, encoding='utf-8')
    print("\n✅ Archivo 'resultados_generativos.csv' generado con éxito. Ábrelo para evaluar los resultados.")


if __name__ == "__main__":
    generar_entregable()