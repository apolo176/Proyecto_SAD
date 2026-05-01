import sys
import pandas as pd
import ollama
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
Clasifica el siguiente texto de una app de música estrictamente en una de estas tres categorías:
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

PROMPT_ONE_SHOT_MEJORADO = """Actúa como un experto en análisis de sentimientos.
Clasifica el sentimiento del texto en A (POSITIVO), B (NEUTRO) o C (NEGATIVO).
REGLA ESTRICTA: Responde ÚNICAMENTE con la letra.

EJEMPLOS:
Texto: "La aplicación va súper fluida y el catálogo es inmenso." -> A
Texto: "Tiene modo oscuro, pero la interfaz es igual a la anterior." -> B
Texto: "Se cierra sola cada dos por tres, es imposible escuchar una canción entera." -> C

Texto a analizar: '{texto}'
Respuesta:"""

# --- EXPERIMENTOS FEW-SHOT ---

PROMPT_FEW_SHOT_MALO = """Actúa como un adolescente en redes sociales. 
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

PROMPT_FEW_SHOT_MEJORADO = """Actúa como un analista de datos evaluando feedback de una app de música.
Clasifica el sentimiento en A (POSITIVO), B (NEUTRO) o C (NEGATIVO). Solo devuelve la letra.

EJEMPLOS:
Texto: "Me encanta la nueva actualización, por fin añadieron letras." -> A
Texto: "Está bien, cumple su función básica de reproducir música." -> B
Texto: "Basura total, me han cobrado dos veces la suscripción." -> C
Texto: "Esperaba más de la interfaz, pero el sonido es espectacular." -> A
Texto: "No entiendo cómo crear listas, el menú es muy confuso." -> C

Texto a analizar: '{texto}'
Respuesta:"""


# ==========================================
# 2. FUNCIÓN DE COMUNICACIÓN CON OLLAMA
# ==========================================

def interactuar_ollama(prompt, modelo="llama3", temperatura=0.0, limite_tokens=15):
    """
    Envía el prompt a Ollama. Si falla, detiene la ejecución inmediatamente.
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
        # Imprimimos el error claramente en consola y detenemos el script
        print("\n" + "=" * 50)
        print("❌ ERROR CRÍTICO AL CONECTAR CON OLLAMA")
        print("=" * 50)
        print(f"Detalle: {e}")
        print(f"💡 Comprobaciones:")
        print(f"  1. ¿Está la aplicación de Ollama ejecutándose en tu PC?")
        print(
            f"  2. ¿Tienes descargado el modelo '{modelo}'? (Prueba a ejecutar 'ollama run {modelo}' en tu terminal).")
        print("\nAbortando la ejecución para no generar un CSV corrupto...")
        sys.exit(1)  # Esto cierra el programa de golpe y devuelve un código de error al sistema


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

    # 3. Tomar una muestra ESTRATIFICADA de DEV para evaluar
    # Usamos en este caso 150 muestras para que no tarde horas, pero manteniendo la proporción de clases
    _, muestra_dev, _, y_muestra_dev = train_test_split(
        X_dev, y_dev,
        test_size=150,
        stratify=y_dev,
        random_state=random_state
    )

    textos_prueba = muestra_dev[columna_texto].tolist()
    y_verdadero = y_muestra_dev.tolist()

    mapa_sentimientos = {'A': 'POSITIVO', 'B': 'NEUTRO', 'C': 'NEGATIVO'}

    experimentos = [
        {"id": "Zero-shot (Malo)", "plantilla": PROMPT_ZERO_SHOT_MALO, "tokens": 20, "mapear": False},
        {"id": "Zero-shot (Normal)", "plantilla": PROMPT_ZERO_SHOT_NORMAL, "tokens": 10, "mapear": False},
        {"id": "Zero-shot (Bueno)", "plantilla": PROMPT_ZERO_SHOT_BUENO, "tokens": 1, "mapear": True},
        {"id": "One-shot (Malo)", "plantilla": PROMPT_ONE_SHOT_MALO, "tokens": 20, "mapear": False},
        {"id": "One-shot (Bueno)", "plantilla": PROMPT_ONE_SHOT_BUENO, "tokens": 1, "mapear": True},
        {"id": "One-shot (Mejorado)", "plantilla": PROMPT_ONE_SHOT_MEJORADO, "tokens": 1, "mapear": True},
        {"id": "Few-shot (Malo)", "plantilla": PROMPT_FEW_SHOT_MALO, "tokens": 15, "mapear": False},
        {"id": "Few-shot (Bueno)", "plantilla": PROMPT_FEW_SHOT_BUENO, "tokens": 1, "mapear": True},
        {"id": "Few-shot (Mejorado)", "plantilla": PROMPT_FEW_SHOT_MEJORADO, "tokens": 1, "mapear": True}
    ]

    modelo_base = "llama3"

    # NUEVO: Estructuras para almacenar todos los datos temporales y puntuaciones
    todos_los_resultados = []
    puntuaciones_f1 = {}

    # --- FASE 1: EVALUACIÓN DE TODOS LOS PROMPTS ---
    for exp in experimentos:
        print(f"\nProbando: {exp['id']}")

        preds_este_experimento = []
        detalles_paso = []

        for texto in tqdm(textos_prueba, desc="Procesando contra Dev"):
            prompt_final = exp['plantilla'].format(texto=texto)
            salida_raw = interactuar_ollama(prompt_final, modelo_base, 0.0, exp['tokens'])

            if exp['mapear']:
                # Limpieza robusta: tomamos el primer caracter válido, aunque se haya limitado con tokens la salida de Ollama.
                letra_limpia = salida_raw.upper().strip().replace(".", "").replace(":", "")[0:1] if salida_raw else ""
                salida_final = mapa_sentimientos.get(letra_limpia, f"ERROR_DE_MAPEO (Raw: {salida_raw})")
                preds_este_experimento.append(salida_final)
            else:
                salida_final = salida_raw.replace("\n", " ")

            detalles_paso.append({
                "id_exp": exp['id'],  # Lo guardamos para la Fase 2
                "Prompt Empleado": prompt_final,
                "Entrada": texto,
                "Salida": salida_final
            })

        # Calcular métrica si el experimento es mapeable
        if exp['mapear'] and len(preds_este_experimento) == len(y_verdadero):
            f1 = f1_score(y_verdadero, preds_este_experimento, average='macro', zero_division=0)
            puntuaciones_f1[exp['id']] = f1
            print(f"📊 F1-Macro obtenido: {f1:.4f}")

        todos_los_resultados.extend(detalles_paso)

    # --- FASE 2: SELECCIÓN AUTOMÁTICA DEL MEJOR PROMPT ---
    if puntuaciones_f1:
        mejor_id = max(puntuaciones_f1, key=puntuaciones_f1.get)
        print(f"\n🏆 El mejor prompt según F1-Macro en DEV es: '{mejor_id}' con {puntuaciones_f1[mejor_id]:.4f}")
    else:
        print("\n⚠️ No se pudieron calcular métricas. Revisa los mapeos.")
        mejor_id = None

    # Formatear la lista final aplicando la etiqueta al ganador
    resultados_finales = []
    for res in todos_los_resultados:
        es_el_mejor = (res['id_exp'] == mejor_id)
        etiqueta = f"[MEJOR RESULTADO] {modelo_base} ({res['id_exp']})" if es_el_mejor else f"{modelo_base} ({res['id_exp']})"

        resultados_finales.append({
            "Modelo y Tamaño": etiqueta,
            "Prompt Empleado": res['Prompt Empleado'],
            "Entrada": res['Entrada'],
            "Salida": res['Salida']
        })

    # Guardar CSV (como ya tenías, asegurando que el mejor va al principio)[cite: 2]
    df_res = pd.DataFrame(resultados_finales)
    df_mejor = df_res[df_res['Modelo y Tamaño'].str.contains(r'\[MEJOR RESULTADO\]', regex=True)]
    df_resto = df_res[~df_res['Modelo y Tamaño'].str.contains(r'\[MEJOR RESULTADO\]', regex=True)]
    df_final = pd.concat([df_mejor, df_resto], ignore_index=True)

    columnas_finales = ["Modelo y Tamaño", "Prompt Empleado", "Entrada", "Salida"]
    df_final[columnas_finales].to_csv("resultados/resultados_generativos.csv", index=False, encoding='utf-8')
    print("\n✅ Archivo 'resultados_generativos.csv' generado con éxito.")


if __name__ == "__main__":
    generar_entregable()