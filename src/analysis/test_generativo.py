#!/usr/bin/env python3
"""
Script de evaluación final para el modelo generativo (Ollama).
Extrae dinámicamente el mejor prompt del CSV generado en la fase anterior,
lo ejecuta sobre el conjunto de test y anexa los resultados.
"""

import os
import sys
import pandas as pd
import ollama
from tqdm import tqdm

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from src.utils.funciones import load_config, load_data, print_section_header

MODELO_OLLAMA = "llama3"


# ==========================================
# 1. FUNCIONES AUXILIARES
# ==========================================

def interactuar_ollama(prompt, modelo=MODELO_OLLAMA, temperatura=0.0, limite_tokens=1):
    """Envía el prompt a Ollama con control de errores estricto."""
    try:
        response = ollama.chat(model=modelo, messages=[
            {'role': 'user', 'content': prompt}
        ], options={
            'temperature': temperatura,
            'num_predict': limite_tokens
        })
        return response['message']['content'].strip()
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ ERROR CRÍTICO AL CONECTAR CON OLLAMA")
        print("=" * 50)
        print(f"Detalle: {e}")
        print("\nAbortando la ejecución...")
        sys.exit(1)


def evaluar_modelo_generativo(y_true, y_pred, nombre_modelo):
    """
    Evalúa las predicciones e imprime las métricas de forma idéntica a test.py.
    Retorna un diccionario con las métricas calculadas.
    """
    print(f"\n{'─' * 70}")
    print(f"📊 Evaluando: {nombre_modelo}")
    print(f"{'─' * 70}")

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n📈 Métricas Globales:")
    print(f"   • Accuracy:        {accuracy:.4f}")
    print(f"   • F1 Macro:        {f1_macro:.4f}")
    print(f"   • F1 Micro:        {f1_micro:.4f}")
    print(f"   • F1 Weighted:     {f1_weighted:.4f}")
    print(f"   • Precision Macro: {precision_macro:.4f}")
    print(f"   • Recall Macro:    {recall_macro:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Matriz de Confusión:")
    print(cm)

    print(f"\n📋 Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, zero_division=0))

    metricas = {
        'modelo': nombre_modelo,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class.tolist()
    }
    return metricas


# ==========================================
# 2. FUNCIÓN PRINCIPAL
# ==========================================

def main():
    print_section_header("PLANTILLA HÍBRIDA - EVALUACIÓN GENERATIVA", char="═")

    # 1. Recuperar el Mejor Prompt Automáticamente
    archivo_resultados_gen = 'resultados/resultados_generativos.csv'
    if not os.path.exists(archivo_resultados_gen):
        print(f"❌ Error: No se encuentra '{archivo_resultados_gen}'.")
        print("💡 Debes ejecutar 'generativo.py' primero para generar este archivo.")
        sys.exit(1)

    print("🔍 Extrayendo el mejor prompt del CSV...")
    df_gen = pd.read_csv(archivo_resultados_gen)
    df_mejor = df_gen[df_gen['Modelo y Tamaño'].str.contains(r'\[MEJOR RESULTADO\]', na=False)]

    if df_mejor.empty:
        print("❌ Error: No se encontró la etiqueta '[MEJOR RESULTADO]' en el CSV.")
        sys.exit(1)

    # Reconstruimos la plantilla sustituyendo la entrada específica por {texto}
    primera_fila = df_mejor.iloc[0]
    prompt_usado = primera_fila['Prompt Empleado']
    entrada_usada = primera_fila['Entrada']
    prompt_ganador = prompt_usado.replace(entrada_usada, "{texto}")

    # --- NUEVA LÓGICA DE EXTRACCIÓN DE NOMBRE ---
    texto_modelo = primera_fila['Modelo y Tamaño']

    # Extraemos desde el primer paréntesis abierto hasta el último cerrado
    # De "ej.: [MEJOR RESULTADO] llama3 (Zero-shot (Bueno))" -> "Zero-shot (Bueno)"
    inicio = texto_modelo.find('(')
    nombre_experimento = texto_modelo[inicio + 1:-1]

    # Formateamos el nombre para que sea un ID limpio (sin espacios ni paréntesis)
    nombre_formateado = nombre_experimento.replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
    nombre_formateado = nombre_formateado.replace('__', '_')  # Por si quedan dobles guiones

    nombre_modelo_gen = f"Ollama_{MODELO_OLLAMA}_{nombre_formateado}"
    # ----------------------------------------------

    print(f"✅ ¡Plantilla '{nombre_experimento}' recuperada con éxito!")

    # 2. Cargar Configuración y Datos de Test
    print("\n📖 Cargando datos de test...")
    config = load_config('config.json', sections=['general', 'test', 'generative'])

    test_file = config.get('test')
    columna_texto = config.get('text_features')[0]
    columna_target = config.get('column')

    if not os.path.exists(test_file):
        print(f"❌ Error: archivo de test '{test_file}' no encontrado")
        sys.exit(1)

    df_test = load_data(test_file)
    print(f"📂 Datos de test cargados: {len(df_test)} instancias en total.")

    eval_test_limit = config.get('eval_test_limit', None)

    if eval_test_limit is not None and eval_test_limit < len(df_test):
        print(f"⚠️ Aplicando límite de test: se evaluarán solo {eval_test_limit} instancias.")
        # Muestreo estratificado para mantener proporciones
        from sklearn.model_selection import train_test_split
        _, df_test_reducido = train_test_split(
            df_test,
            test_size=eval_test_limit,
            stratify=df_test[columna_target],
            random_state=config['general']['random_state']
        )
        df_test = df_test_reducido
    else:
        print("📊 Evaluando sobre TODAS las instancias de test.")

    textos_test = df_test[columna_texto].tolist()
    y_verdadero = df_test[columna_target].tolist()
    y_verdadero = [str(y).upper().strip() for y in y_verdadero]

    mapa_sentimientos = {'A': 'POSITIVO', 'B': 'NEUTRO', 'C': 'NEGATIVO'}
    predicciones = []

    # 3. Inferencia con Ollama
    print(f"\n🤖 Iniciando inferencia en Test con el modelo: {nombre_modelo_gen}")
    for texto in tqdm(textos_test, desc="Evaluando TEST"):
        prompt_final = prompt_ganador.format(texto=texto)
        salida_raw = interactuar_ollama(prompt_final, limite_tokens=1)

        letra = salida_raw.upper().strip().replace(".", "").replace(":", "")[0:1] if salida_raw else ""
        salida_final = mapa_sentimientos.get(letra, "ERROR_MAPEO")
        predicciones.append(salida_final)

    # 4. Evaluación
    metricas_generativo = evaluar_modelo_generativo(y_verdadero, predicciones, nombre_modelo_gen)

    # 5. Integración con resultados de modelos tradicionales
    print_section_header("INTEGRACIÓN DE RESULTADOS", char="─")
    metricas_output = config.get('metricas_output', 'resultados/metricas_modelos.csv')
    df_nuevo = pd.DataFrame([metricas_generativo])

    if os.path.exists(metricas_output):
        df_existente = pd.read_csv(metricas_output)
        df_existente = df_existente[df_existente['modelo'] != nombre_modelo_gen]
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    else:
        print("⚠️ No se encontró archivo previo de modelos tradicionales.")
        df_final = df_nuevo

    df_final = df_final.sort_values('f1_macro', ascending=False)
    os.makedirs('resultados', exist_ok=True)
    df_final.to_csv(metricas_output, index=False)
    print(f"💾 Métricas actualizadas guardadas en: {metricas_output}")

    print(f"\n🏆 RANKING GLOBAL ACTUALIZADO (Tradicionales vs Generativo):")
    print(f"{'─' * 70}")
    for i, row in df_final.iterrows():
        print(f"{row['modelo']:30s} | F1 Macro: {row['f1_macro']:.4f} | Accuracy: {row['accuracy']:.4f}")
    print(f"{'─' * 70}")


if __name__ == '__main__':
    main()