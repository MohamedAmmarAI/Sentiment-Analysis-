import streamlit as st
import torch
import joblib
import whisper
import re
import pandas as pd
import emoji
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
from qalsadi.lemmatizer import Lemmatizer
import nltk

# Download stopwords if not already available
nltk.download("stopwords")

# Load models
vectorizer = joblib.load('vectorizer.joblib')
logistic_model = joblib.load('logistic_model.joblib')

# Load Whisper model
whisper_model = whisper.load_model("large")

# Load custom BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-mini-arabic")

class CustomBertClassifier(torch.nn.Module):
    def __init__(self, num_classes=3, model_name="asafaya/bert-mini-arabic"):
        super(CustomBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embeddings)

# Initialize BERT model
bert_model = CustomBertClassifier()

# Load state_dict with key renaming to match the model structure
try:
    # Load the saved state_dict
    state_dict = torch.load('bert_classifier.pth')
    
    # Rename keys if necessary
    updated_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("new_classifier", "classifier")  # Adjust if necessary
        updated_state_dict[new_key] = value
    
    # Load the updated state_dict
    bert_model.load_state_dict(updated_state_dict)
except RuntimeError as e:
    # Fallback: Load state_dict with strict=False if there’s still a mismatch
    st.warning("Partial state_dict loaded due to mismatch, using strict=False as fallback.")
    bert_model.load_state_dict(torch.load('bert_classifier.pth'), strict=False)

# Set model to evaluation mode
bert_model.eval()

# Define text preprocessing
stop_words = set(stopwords.words("arabic"))
lemmer = Lemmatizer()
emojis = {
    "🙂": "يبتسم", "😂": "يضحك", "💔": "قلب حزين", "❤️": "حب", "😭": "يبكي", "😢": "حزن", "😔": "حزن", "😄": "يضحك"
}

def returnCleanText(text):
    # Remove punctuation
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~@#"""), ' ', text)
    text = text.replace('؛', "")
    return text

# Streamlit App Interface
st.title("Arabic Sentiment Analysis from Audio")

st.write("Upload an audio file in Arabic to analyze its sentiment.")
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "mpeg"])

if uploaded_file is not None:
    audio_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Transcribe audio
    with st.spinner("Transcribing audio..."):
        transcription = whisper_model.transcribe(audio_path, language="ar")
        text = transcription["text"]
    
    st.write("Transcribed Text:")
    st.write(text)

    clean_text = returnCleanText(text)

    # Logistic Regression Prediction
    feature_text = vectorizer.transform([clean_text])
    logistic_pred = logistic_model.predict(feature_text)[0]
    sentiment_logistic = {0: "Negative", 1: "Neutral", 2: "Positive"}[logistic_pred]

    # BERT Model Prediction
    tokenized_text = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        bert_output = bert_model(tokenized_text["input_ids"], attention_mask=tokenized_text["attention_mask"])
        bert_pred = torch.argmax(bert_output, dim=1).item()
    sentiment_bert = {0: "Negative", 1: "Neutral", 2: "Positive"}[bert_pred]

    # Display Results
    st.write("Predicted Sentiment (Logistic Regression):")
    st.write(sentiment_logistic)

    st.write("Predicted Sentiment (BERT Model):")
    st.write(sentiment_bert)
