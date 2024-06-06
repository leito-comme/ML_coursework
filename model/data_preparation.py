import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['review_text', 'review_rating']].dropna(subset=['review_text', 'review_rating'])
    data['review_text'] = data['review_text'].astype(str)
    reviews = data["review_text"].values
    ratings = data["review_rating"].values
    ratings = ratings.astype(int)

    label_encoder = LabelEncoder()
    ratings = label_encoder.fit_transform(ratings)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    file_path = "../resources/dataset.csv"
    X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(file_path)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
