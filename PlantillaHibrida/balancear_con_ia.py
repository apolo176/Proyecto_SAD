import pandas as pd
import ollama
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from funciones import load_config, load_data

PROMPT_PARAFRASIS = """Reescribe la siguiente opinión de una app de citas manteniendo exactamente el MISMO sentimiento, pero usando diferentes palabras y estructuras.
No añadas introducciones, confirmaciones ni comillas. Genera SOLO la reseña.

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
        test_size=config['preprocessing']['test_size'],
        stratify=df[col_target],
        random_state=config['general']['random_state']
    )

    # Unimos temporalmente X_train e y_train para manipularlos
    df_train = pd.concat([X_train, y_train], axis=1)

    conteo = df_train[col_target].value_counts()
    clase_minoritaria = conteo.idxmin()
    print(f"📊 Clase minoritaria detectada en TRAIN: '{clase_minoritaria}' con {conteo.min()} muestras.")

    # 3. Coger los textos base solo de TRAIN
    textos_base = df_train[df_train[col_target] == clase_minoritaria][col_texto].head(50).tolist()

    nuevas_filas = []
    print(f"🧠 Generando {len(textos_base)} nuevas opiniones de tipo '{clase_minoritaria}'...")

    for texto in tqdm(textos_base):
        prompt_final = PROMPT_PARAFRASIS.format(texto=texto)

        respuesta = ollama.chat(
            model="llama3",
            messages=[{'role': 'user', 'content': prompt_final}],
            options={'temperature': 0.8, 'num_predict': 100}
        )
        nueva_review = respuesta['message']['content'].strip()

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