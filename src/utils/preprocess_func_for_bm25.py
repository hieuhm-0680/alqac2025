from typing import List
import re
import nltk

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


from nltk.tokenize import word_tokenize


def tokenize_text(text):
    ensure_nltk_resources()
    return word_tokenize(text)


def preprocess_func_for_bm25(text: str) -> List[str]:
    # Remove \xAD
    text = text.replace('\xAD', '')

    # Remove escape characters \n, \t, \r
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # Replace links with <url>
    text = re.sub(r'http\S+|www\.\S+', '<url>', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Lower text
    text = text.lower()
    
    return tokenize_text(text)