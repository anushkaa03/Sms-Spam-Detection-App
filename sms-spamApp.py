import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data



nltk_data_path = "/usr/local/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Ensure required NLTK resources are downloaded
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
stopwords.words('english')
# Initialize objects
ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    # Keep only alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Perform stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)


# Load vectorizer and model
import pickle

# Load vectorizer and model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Title and introduction
st.title("📱 SMS SPAM DETECTION MODEL")
st.write("Made by Anushka Ojha 🚀")

# User input section
st.write("### Enter the SMS below:")
input_sms = st.text_input("Type your SMS")

# Initialize result variable to handle cases when input is empty
result = None

# Button in a separate container
predict_button = st.button('Predict')

if predict_button:
    if not input_sms.strip():  # Check if the input is empty or contains only whitespace
        st.warning("⚠️ Please enter a valid SMS before clicking Predict.")
    else:
        with st.spinner('Processing...'):
            transformed_sms = transform_text(input_sms)
            vector_input = tk.transform([transformed_sms])
            result = model.predict(vector_input)[0]  # Predict the result

    # Show result in a centered layout only if 'result' is defined
    if result is not None:
        if result == 1:
            st.header("🚨 SPAM 🚨")
        else:
            st.header("✅ NOT SPAM ✅")
