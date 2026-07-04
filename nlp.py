import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text=text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)


    text = y[:]
    y.clear() # it will clear the list with containing alphanumeric words and move to next process

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear() # it will clear the list with containing alphanumeric words and move to next process

    for i in text:
        y.append(ps.stem(i))

    return" ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Text Fraud Detector using Machine Learning Model by Shamim Khan")

input_text = st.text_area("Enter your message:")


if st.button('Predict'):

    # 1. Preprocess
    transformed_data = transform_text(input_text)
    # 2. Vectorization
    vector_input = tfidf.transform([transformed_data])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")