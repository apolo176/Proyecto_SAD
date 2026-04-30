import pandas as pd
import re
import argparse
from collections import Counter
from nltk.corpus import stopwords

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',help='CSV de entrada', required=True, type=str)
parser.add_argument('-o','--output',help='CSV de salida', required=True, type=str)
args = parser.parse_args()


df = pd.read_csv(args.input)
df = df.dropna(subset=['review'])

nltk_stopwords = set(stopwords.words('english'))

custom_stopwords = {'the', 'and', 'to', 'a', 'of', 'in', 'it', 'is', 'i', 'for', 
                    'this', 'that', 'with', 'on', 'my', 'app', 'music', 'song',
                    'songs', 'just', 'like', 'get', 'but', 'have', 'so', 'as',
                    'not', 'cant', 'dont', 'didnt', 'wont', 'isnt', 'even',
                    'when', 'very', 'doesnt', 'from', 'want', 'using',
                    'really', 'keeps', 'every', 'makes', 'also', 'would', 'make',
                    'used', 'know', 'keep', 'since', 'says', 'thats', 'thing',
                    'something', 'cannot', 'sometimes', 'overall', 'things', 
                    'first', 'next', 'though', 'feels', 'feel', 'second', 'last',
                    'lastly', 'star', 'fine', 'theres', 'thank', 'couldnt',
                    'wanted', 'needs', 'good', 'great', 'excellent', 'amazing',
                    'awesome', 'terrible', 'horrible', 'bad', 'worst', 'best',
                    'love', 'hate', 'nice', 'perfect', 'useless', 'better',
                    'ever', 'never', 'need', 'tried', 'well', 'much', 'annoying',
                    'frustrating', 'sucks', 'seems', 'trash', 'poor', 'gets',
                    'worse', 'properly', 'part', 'anymore','cool','many','time',
                    'please', 'always', 'still', 'enjoy', 'everything', 'give',
                    'could', 'problem', 'another', 'times', 'super', 'going',
                    'choose', 'change', 'able', 'liked', 'different', 'think', 
                    'absolutely', 'definitely', 'right', 'nothing', 'anything',
                    'thanks', 'honestly', 'point', 'instead', 'take', 'personally',
                    'wish', 'favourite', 'guys', 'actually', 'switching',
                    'stands', 'making', 'fantastic', 'already', 'youre',
                    'enjoyable', 'wonderful', 'happy', 'pretty', 'yall',
                    'specific', 'almost', 'exceptional', 'exceptionally', 'whats',
                    'especially', 'else', 'whole', 'everytime', 'beautiful',
                    'giving', 'hope'}

stopwords = nltk_stopwords.union(custom_stopwords)

mapa_sentimiento = {'positivo': 1, 'negativo': -1}
df['valor_sent'] = df['Polaridad_Clustering'].map(mapa_sentimiento).fillna(0)

palabras_data = []

for _, row in df.iterrows():
    texto = re.sub(r'[^a-z\s]', '', str(row['review']).lower())
    palabras = [p for p in texto.split() if p not in stopwords and len(p) > 3]
    
    for p in palabras:
        palabras_data.append({'Palabra': p, 'Sentimiento_Contexto': row['valor_sent']})

df_palabras = pd.DataFrame(palabras_data)

df_final = df_palabras.groupby('Palabra').agg(
    Frecuencia=('Palabra', 'count'),
    Sentimiento_Promedio=('Sentimiento_Contexto', 'mean')
).reset_index()

df_final = df_final.sort_values('Frecuencia', ascending=False).head(150)

df_final.to_csv(args.output, index=False)
print(f"{args.output} generado con éxito.")
