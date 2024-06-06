def train_model(model, X_train_pad, y_train, epochs=12, batch_size=64):
    # Train the model
    history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return history

if __name__ == "__main__":
    from model_building import build_model
    from text_preprocessing import preprocess_text
    from data_preparation import load_and_prepare_data

    file_path = "../resources/dataset.csv"
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(file_path)
    X_train_pad, X_test_pad, tokenizer, vocab_size = preprocess_text(X_train, X_test)

    model = build_model(vocab_size, 200)
    history = train_model(model, X_train_pad, y_train)
