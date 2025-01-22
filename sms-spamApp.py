import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize objects
ps = PorterStemmer()

# Function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# Load vectorizer and model
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
