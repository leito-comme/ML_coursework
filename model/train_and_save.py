import os
from tensorflow.keras.models import load_model
from model_training import train_model
from model_building import build_model
from text_preprocessing import preprocess_text
from data_preparation import load_and_prepare_data

def train_and_save_model(file_path, model_save_path):
    X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(file_path)
    X_train_pad, X_test_pad, tokenizer, vocab_size = preprocess_text(X_train, X_test)
    
    model = build_model(vocab_size, 200)
    train_model(model, X_train_pad, y_train)
    
    # Save model and tokenizer
    model.save(model_save_path)
    tokenizer_path = model_save_path.replace('.h5', '_tokenizer.pkl')
    label_encoder_path = model_save_path.replace('.h5', '_label_encoder.pkl')

    with open(tokenizer_path, 'wb') as f:
        import pickle
        pickle.dump(tokenizer, f)
    
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    file_path = "../resources/dataset.csv"
    model_save_path = "../resources/review_rating_model.h5"
    train_and_save_model(file_path, model_save_path)
