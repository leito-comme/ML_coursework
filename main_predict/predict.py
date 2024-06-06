import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle


def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text


def load_resources(model_path):
    model = load_model(model_path)
    tokenizer_path = model_path.replace(".h5", "_tokenizer.pkl")
    label_encoder_path = model_path.replace(".h5", "_label_encoder.pkl")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


def predict_rating(model, review_text, tokenizer, max_length, label_encoder):
    review_text = clean_text(review_text)
    review_seq = tokenizer.texts_to_sequences([review_text])
    review_pad = pad_sequences(review_seq, maxlen=max_length, padding="post")
    predicted_rating = model.predict(review_pad)
    predicted_rating = label_encoder.inverse_transform([np.argmax(predicted_rating)])
    return predicted_rating[0]



def mendelbrot(reviews: list):
    # Flatten and clean reviews to obtain a list of all words
    all_words = [word for review in reviews for word in clean_text(review['review']).split()]
    word_freq = Counter(all_words)
    res = []
    diagram_data = []
    
    for word, freq in word_freq.items():
        temp = math.fabs(math.log(freq + 1, 2))
        res.append(temp)
        if temp >= 0.5:
            diagram_data.append({word: temp})
    
    # print(f"Naturalness of language: {round(sum(res) / len(res), 5)}")
    
    diagram_data = diagram_data[:10]
    _, ax = plt.subplots()
    ax.pie(
        [value for item in diagram_data for _, value in item.items()],
        labels=[key for item in diagram_data for key, _ in item.items()],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.grid()
    plt.show()


def zipf_first_law(reviews: list):
    counter = {
        k: v
        for k, v in sorted(
            dict(
                Counter([word for item in reviews for word in item["review"].split()])
            ).items(),
            key=lambda x: x[1],
        )
    }

    keys_length = {}

    for key in counter.keys():
        keys_length[key] = len(key)

    sorted_key_length = {
        k: v for k, v in sorted(keys_length.items(), key=lambda item: item[1])
    }

    plt.title("Zipf's first law")
    plt.xlabel("Meaning of the word")
    plt.ylabel("Frequency")
    plt.plot(
        list(sorted_key_length.values()),
        [counter[key] for key in list(sorted_key_length.keys())],
    )
    plt.show()


def zipf_second_law(reviews: list):
    counter = dict(
        Counter([word for item in reviews for word in item["review"].split()])
    )

    lengths = [length for length in range(len(counter))]

    plt.title("Zipf's second law")
    plt.xlabel("Frequency")
    plt.ylabel("Word count")
    plt.plot(lengths, list(counter.values()))
    plt.show()


if __name__ == "__main__":
    model_path = "../resources/review_rating_model.h5"
    model, tokenizer, label_encoder = load_resources(model_path)

    reviews = [
        "Nice materials, good quality",
        "Strange staff, I don't like service",
        "We don't like this shop to be honest",
        "Cool stuff, everything was perfect",
        "Quite good mag, but not too much",
        "Insane and gorgeous prices, highly recommend",
        "Excellent customer service and fast delivery",
        "Product quality is average, not worth the price",
        "Amazing selection of items, will buy again",
        "Terrible experience, won't shop here again",
        "Love the variety, but some items are overpriced",
        "Quick delivery but the packaging was damaged",
        "Fantastic deals and great customer support",
        "The website is user-friendly and easy to navigate",
        "Disappointed with the product, not as described",
        "Great shopping experience, will recommend to friends",
        "Received the wrong item, very frustrating",
        "Affordable prices but quality could be better",
        "Super fast shipping and excellent quality products",
        "Customer service was unhelpful and rude",
        "High quality products at reasonable prices",
        "Not satisfied with the purchase, poor quality",
        "Beautiful items, just as pictured",
        "Very slow shipping, won't order again",
        "Extremely happy with my purchase, highly recommend",
        "Good prices but slow delivery",
        "Lovely shop, very satisfied with the products",
        "Items arrived damaged, very disappointed",
        "Best shopping experience, fantastic products",
        "Fast delivery and great communication",
        "Products are not as advertised, very disappointed",
        "Great prices and excellent quality",
        "Poor customer service, very rude staff",
        "Satisfied with the purchase, will shop again",
        "Too expensive for the quality offered",
        "Highly recommend this shop, very pleased",
        "Shipping took forever, not happy",
        "Wonderful products and quick service",
        "Mediocre quality, not worth the price",
        "User-friendly website and quick checkout",
        "Received my order quickly, very pleased",
        "Packaging was damaged, but product was fine",
        "Amazing products, highly recommend",
        "Customer service is not responsive",
        "Good experience overall, will shop again",
        "Quality doesn't match the description",
        "Great value for money, very happy",
        "Products arrived late, not satisfied",
        "Excellent shopping experience, thank you",
        "Not impressed with the product quality",
        "Fast and reliable shipping, great service",
        "The items were just as described, very happy",
        "Customer service resolved my issue quickly",
        "Prices are too high for the quality",
        "Great product selection and good prices",
        "Shipping was fast, but items were damaged",
        "Very pleased with my purchase, thank you",
        "Poor quality and terrible customer service",
        "Great shop with excellent products",
        "Shipping took too long, not happy",
        "Quality exceeded my expectations, very happy",
        "Easy to navigate website, smooth transaction",
        "Items arrived on time and in good condition",
        "Customer service was very helpful",
        "Very satisfied with my purchase",
        "Product quality is not as good as expected",
        "Shipping was delayed, but worth the wait",
        "Great deals and fast delivery",
        "Poor packaging, items were damaged",
        "Excellent quality products and good service",
        "Will definitely shop here again",
        "Not worth the money, very disappointed",
        "Good prices, but slow shipping",
        "Amazing customer service, very helpful",
        "Products arrived quickly and in perfect condition",
        "Very bad experience, won't shop again",
        "High quality items and fast shipping",
        "Website was easy to use, very convenient",
        "Items not as described, very disappointed",
        "Fantastic shop, very happy with my purchase",
        "Shipping took too long, not impressed",
        "Great quality products and fast delivery",
        "Customer service was not helpful at all",
        "Items arrived on time, very happy",
        "Not satisfied with the product quality",
        "Affordable prices and good quality",
        "Received my order quickly, very pleased",
        "Items were damaged during shipping",
        "Great selection of products, very happy",
        "Customer service was very rude",
        "Will not be shopping here again",
        "Excellent quality and fast shipping",
        "Very pleased with the service",
        "Product quality is subpar, not happy",
        "Smooth transaction and fast delivery",
        "Items arrived late, very disappointed",
        "Great customer service and quality products",
        "Shipping was fast, very pleased",
        "Products are of low quality, not happy",
        "Good experience, will shop again",
        "Packaging was poor, items were damaged",
        "Amazing deals and fast shipping",
        "Customer service was very helpful",
        "Quality is not as expected, disappointed",
        "Very satisfied with my purchase",
        "Shipping took too long, not happy",
        "Great quality and fast delivery",
        "Poor customer service, very rude",
        "Received my order quickly and in good condition",
        "Items were not as described, disappointed",
        "Great prices and fast shipping",
        "Customer service resolved my issue quickly",
        "Quality is not as advertised, very disappointed",
        "Very happy with my purchase, thank you",
        "Shipping was fast and reliable",
        "Poor quality and terrible customer service",
        "Great product selection and good prices",
        "Items arrived on time, very pleased",
        "Not impressed with the quality",
        "Excellent customer service and fast delivery",
        "Shipping was delayed, but worth the wait",
        "Good experience overall, will shop again",
        "Product quality is average, not worth the price",
        "Items were damaged during shipping",
        "Great deals and fast delivery",
        "Very satisfied with the purchase",
        "Poor packaging, items were damaged",
        "Fantastic products and quick service",
        "Not worth the money, very disappointed",
        "Quality exceeded my expectations, very happy",
        "Easy to navigate website and quick checkout",
        "Shipping took too long, not impressed",
        "Great prices and excellent quality",
        "Received the wrong item, very frustrating",
        "Satisfied with the purchase, will shop again",
        "Products are not as advertised, disappointed",
        "Customer service was unhelpful and rude",
        "Amazing products, highly recommend",
        "Shipping was fast, very pleased",
        "Quality is not as expected, disappointed",
        "Great value for money, very happy",
        "Items arrived late, very disappointed",
        "Fast delivery and great communication",
        "Products arrived quickly and in perfect condition",
        "Very pleased with the service",
        "Affordable prices but quality could be better",
        "Shipping took forever, not happy",
        "Beautiful items, just as pictured",
        "Customer service is not responsive",
        "Very bad experience, won't shop again",
        "Items arrived on time and in good condition",
        "Not satisfied with the product quality",
        "Wonderful products and quick service",
        "Products are of low quality, not happy",
        "User-friendly website and quick checkout",
        "Packaging was damaged, but product was fine",
        "Excellent quality products and good service",
        "Shipping was delayed, not happy",
        "Very happy with my purchase, thank you",
        "Poor quality and terrible customer service",
        "Customer service was very helpful",
        "Great quality products and fast delivery",
        "Not worth the money, very disappointed",
        "Customer service resolved my issue quickly",
        "Good prices but slow delivery",
        "Items not as described, very disappointed",
        "Products arrived late, not satisfied",
        "High quality items and fast shipping",
        "Not satisfied with the product quality",
        "Shipping took too long, not happy",
        "Fantastic shop, very happy with my purchase",
        "Poor packaging, items were damaged",
        "Items arrived quickly, very pleased",
        "Quality is not as good as expected",
        "Very pleased with my purchase",
        "Shipping was fast, very pleased",
        "Products arrived on time, very happy",
        "Great deals and fast delivery",
        "Customer service was not helpful at all",
        "Received my order quickly and in good condition",
        "Not impressed with the product quality",
        "Affordable prices and good quality",
        "Website was easy to use, very convenient",
        "Poor customer service, very rude",
        "Great selection of products, very happy",
        "Received the wrong item, very frustrating",
        "Great customer service and quality products",
        "Items were damaged during shipping",
        "Smooth transaction and fast delivery",
        "Quality is not as advertised, very disappointed",
    ]
    predicted_rating = [
        predict_rating(model, review, tokenizer, 200, label_encoder)
        for review in reviews
    ]
    reviews_object = []

    for i in range(len(reviews)):
        reviews_object.append({"review": reviews[i], "score": predicted_rating[i]})
        print("Review: " + reviews[i])
        print("Predicted rating: ", predicted_rating[i], "\n")

    mendelbrot(reviews_object)
    zipf_first_law(reviews_object)
    zipf_second_law(reviews_object)
