import re
import string

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Завантаження стоп-слів англійської мови
stop_words = set(stopwords.words("english"))

# Ініціалізація лематизатора WordNet
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """
    Перетворює тег частини мови Penn Treebank у відповідний тег WordNet.

    Parameters
    ----------
    treebank_tag : str
        Тег частини мови у форматі Penn Treebank.

    Returns
    -------
    str
        Відповідний тег WordNet для лематизації.
    """

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text):
    """
    Виконує попередню обробку тексту.

    Функція переводить текст у нижній регістр, видаляє HTML-теги,
    пунктуацію та стоп-слова, виконує токенізацію, визначення
    частин мови і лематизацію.

    Parameters
    ----------
    text : str
        Вхідний текст для попередньої обробки.

    Returns
    -------
    str
        Очищений та лематизований текст.
    """

    # Переведення тексту в нижній регістр
    text = text.lower()

    # Видалення HTML-тегів
    text = re.sub(r"<.*?>", " ", text)

    # Видалення пунктуації
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Токенізація тексту
    tokens = word_tokenize(text)

    # Залишаємо лише слова та видаляємо стоп-слова
    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    # Визначення частин мови
    tagged_tokens = pos_tag(tokens)

    # Лематизація слів
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]

    return " ".join(lemmatized)