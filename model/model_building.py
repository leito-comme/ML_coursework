from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional

def build_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # Output layer with 5 classes (ratings 1 to 5)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    from text_preprocessing import preprocess_text
    from data_preparation import load_and_prepare_data

    file_path = "../resources/dataset.csv"
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(file_path)
    X_train_pad, X_test_pad, tokenizer, vocab_size = preprocess_text(X_train, X_test)
    
    model = build_model(vocab_size, 200)
    model.summary()
