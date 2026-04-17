import pandas as pd
import ollama
from tqdm import tqdm

PROMPT_PARAFRASIS = """Reescribe la siguiente opinión de una app de citas manteniendo exactamente el MISMO sentimiento, pero usando diferentes palabras y estructuras.
No añadas introducciones, confirmaciones ni comillas. Genera SOLO la reseña.

Original: '{texto}'
Reescritura:"""


def balancear_dataset():
    print("🤖 Iniciando Oversampling Generativo...")

    # 1. Cargar datos
    df = pd.read_csv('data/train.csv')
    conteo = df['sentiment'].value_counts()
    clase_minoritaria = conteo.idxmin()
    print(f"📊 Clase minoritaria detectada: '{clase_minoritaria}' con {conteo.min()} muestras.")

    # 2. Coger los textos a multiplicar (por ejemplo, cogemos los primeros 50 para no tardar mucho hoy)
    textos_base = df[df['sentiment'] == clase_minoritaria]['review_text'].head(50).tolist()

    nuevas_filas = []

    print(f"🧠 Generando {len(textos_base)} nuevas opiniones de tipo '{clase_minoritaria}'...")
    for texto in tqdm(textos_base):
        prompt_final = PROMPT_PARAFRASIS.format(texto=texto)

        # Llamar a Llama3 con temperatura 0.8 para que sea creativo inventando
        respuesta = ollama.chat(
            model="llama3",
            messages=[{'role': 'user', 'content': prompt_final}],
            options={'temperature': 0.8, 'num_predict': 100}
        )
        nueva_review = respuesta['message']['content'].strip()

        nuevas_filas.append({'review_text': nueva_review, 'sentiment': clase_minoritaria})

    # 3. Unir y guardar (SIN DATA LEAKAGE)
    df_nuevas = pd.DataFrame(nuevas_filas)
    df_nuevas.to_csv('resultados/parafrases_generadas_ollama.csv', index=False, encoding='utf-8')
    print("📁 Archivo de entregable generado: 'resultados/parafrases_generadas_ollama.csv'")
    df_balanceado = pd.concat([df, df_nuevas], ignore_index=True)

    df_balanceado.to_csv('data/train_balanceado_ia.csv', index=False, encoding='utf-8')
    df_balanceado = pd.concat([df, df_nuevas], ignore_index=True)

    df_balanceado.to_csv('data/train_balanceado_ia.csv', index=False, encoding='utf-8')
    print("\n🎉 ¡Dataset balanceado guardado en 'data/train_balanceado_ia.csv'!")
    print("Nuevo conteo de clases:")
    print(df_balanceado['sentiment'].value_counts())


if __name__ == "__main__":
    balancear_dataset()