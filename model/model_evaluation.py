from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test_pad, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Predictions
    y_pred = model.predict(X_test_pad)
    y_pred_classes = y_pred.argmax(axis=1)

    # Classification Report
    print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

    # Confusion Matrix
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

    return loss, accuracy

if __name__ == "__main__":
    from model_training import train_model
    from model_building import build_model
    from text_preprocessing import preprocess_text
    from data_preparation import load_and_prepare_data

    file_path = "../resources/dataset.csv"
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(file_path)
    X_train_pad, X_test_pad, tokenizer, vocab_size = preprocess_text(X_train, X_test)

    model = build_model(vocab_size, 200)
    train_model(model, X_train_pad, y_train)
    evaluate_model(model, X_test_pad, y_test)
