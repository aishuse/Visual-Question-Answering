import streamlit as st
import numpy as np
import cv2
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# Load model, tokenizer, and label encoder
@st.cache_resource
def load_all():
    model = load_model('full_model_new_24.keras')
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    labelencoder = pickle.load(open('labelencoder.pkl', 'rb'))
    return model, tokenizer, labelencoder

model, tokenizer, labelencoder = load_all()

# Decontraction helper
def decontractions(phrase):
    phrase = re.sub(r"won['’]t", "will not", phrase)
    phrase = re.sub(r"can['’]t", "can not", phrase)
    phrase = re.sub(r"he['’]s", "he is", phrase)
    phrase = re.sub(r"she['’]s", "she is", phrase)
    phrase = re.sub(r"it['’]s", "it is", phrase)
    phrase = re.sub(r"n['’]t", " not", phrase)
    phrase = re.sub(r"['’]re", " are", phrase)
    phrase = re.sub(r"['’]d", " would", phrase)
    phrase = re.sub(r"['’]ll", " will", phrase)
    phrase = re.sub(r"['’]ve", " have", phrase)
    phrase = re.sub(r"['’]m", " am", phrase)
    return phrase

# Text preprocessing
def text_preprocess(text):
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[-,:]', ' ', text)
    text = re.sub("(?!<=\d)(\.)(?!\d)", '', text)
    text = re.sub('[^A-Za-z0-9. ]+', '', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

# Title
st.title("🧠 Visual Question Answering")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
question = st.text_input("Ask a question about the image")

if uploaded_file and question:
    # Read and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Preprocess question
    question_clean = text_preprocess(question)
    question_seq = pad_sequences(tokenizer.texts_to_sequences([question_clean]), maxlen=22, padding='post')

    # Predict
    pred_probs = model.predict([img_input, question_seq], verbose=0)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    predicted_answer = labelencoder.inverse_transform([pred_class])[0]

    st.subheader("Predicted Answer:")
    st.success(predicted_answer)
