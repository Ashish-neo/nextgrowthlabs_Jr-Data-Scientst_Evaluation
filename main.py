import streamlit as st
import numpy as np
import pandas as pd
import textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords




port = PorterStemmer()

def text_cleaner(text):
    cleaned = re.sub('[^a-zA-Z]', " ", text)
    cleaned = cleaned.lower()
    cleaned = cleaned.split()
    cleaned = [port.stem(word) for word in cleaned if word not in stopwords.words("english")]
    cleaned = ' '.join(cleaned)
    return cleaned

st.title("Identifying Incorrect Ratings")
st.header("Instructions")
st.markdown("1.Review column's name should be **Text**")
st.markdown("2.Rating column's name should be **Star**")
st.markdown("3.Rating range should be 0-5")

uploaded_file = st.file_uploader(label="Choose a File",
                                 type=['csv','xlsx'])


df = pd.read_csv(uploaded_file)
st.dataframe(df)

if st.button("Click for Results") :
    df["Cleaned_Text"] = df["Text"].apply(lambda x: text_cleaner(str(x)))

    sid = SentimentIntensityAnalyzer()

    df["Vader_Score"] = df["Cleaned_Text"].apply(lambda review:sid.polarity_scores(review))
    df["Vader_Compound_Score"]  = df['Vader_Score'].apply(lambda score_dict: score_dict['compound'])
    df["Result"] = df["Vader_Compound_Score"].apply(lambda c: 'positive' if c > 0 else ('negative' if c < 0 else 'neutral'))

    df_focus = df[(df.Result == "positive")]
    df_focus["Suggestion"] =  df_focus["Star"].apply(lambda star: "No Focus Needed" if star >= 3 else "Attention Needed")

    keyword = ['good', 'nice', 'thank you', 'best', 'awesome', 'helpful']

    final =  df_focus[( df_focus["Suggestion"] == "Focus Needed")]
    final = final[final["Cleaned_Text"].isin(keyword)]

    display_df = final[['Text','Star','Cleaned_Text','Suggestion']]

    st.dataframe(display_df)

    st.bar_chart(df_focus.Suggestion.value_counts())

    data = final

    st.download_button(
        label="Download data as CSV",
        data=data.to_csv().encode("utf-8"),
        file_name='data.csv',
        mime='text/csv',
    )
