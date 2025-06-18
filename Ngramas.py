from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

def obtener_ngrams(corpus, ngram_range=(2,2), min_df=1):
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    X = vectorizer.fit_transform(corpus)
    ngrams = vectorizer.get_feature_names_out()
    frecuencias = X.toarray().sum(axis=0)
    return ngrams, frecuencias

def graficar_ngrams(ngrams, frecuencias, top=10, titulo=""):
    indices = frecuencias.argsort()[::-1][:top]
    ngrams_top = [ngrams[i] for i in indices]
    frecuencias_top = frecuencias[indices]
    plt.figure(figsize=(10,6))
    plt.barh(ngrams_top[::-1], frecuencias_top[::-1])
    plt.xlabel("Frecuencia")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

def main():
    #force la ruta, porque no me leia el corpus solo llamandolo por su nombre
    ruta_corpus = r"C:\Users\Jaz\Desktop\Practicas\2M\CorpusEducacion.txt"
    with open(ruta_corpus, encoding="UTF-8") as f:
        texto = f.read()

    texto_limpio = preprocess_text(texto)
    corpus_final = [texto_limpio]

    # Gráfico de 2-gramas
    ngrams2, frec2 = obtener_ngrams(corpus_final, ngram_range=(2,2), min_df=1)
    graficar_ngrams(ngrams2, frec2, top=10, titulo="Top 10 2-gramas")

    # Gráfico de 3-gramas
    ngrams3, frec3 = obtener_ngrams(corpus_final, ngram_range=(3,3), min_df=1)
    graficar_ngrams(ngrams3, frec3, top=10, titulo="Top 10 3-gramas")

if __name__ == "__main__":
    main()
