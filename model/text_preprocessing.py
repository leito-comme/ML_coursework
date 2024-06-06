from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def preprocess_text(X_train, X_test, max_words=10000, max_length=200):
    X_train = [clean_text(text) for text in X_train]
    X_test = [clean_text(text) for text in X_test]

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

    vocab_size = len(tokenizer.word_index) + 1  

    return X_train_pad, X_test_pad, tokenizer, vocab_size

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data

    file_path = "../resources/dataset.csv"
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(file_path)
    X_train_pad, X_test_pad, tokenizer, vocab_size = preprocess_text(X_train, X_test)
    print("Vocabulary size:", vocab_size)
