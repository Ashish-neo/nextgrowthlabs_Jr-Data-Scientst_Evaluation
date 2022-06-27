import streamlit as st
import numpy as np
import pandas as pd

from textblob import TextBlob


st.title("Identifying Incorrect Ratings")
st.header("Bad Review")


file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if file is not None:
    data = pd.read_csv(file)
else:
    st.text("Please upload a csv file")

if st.button("Click for Results"):
    df = data[['Text', 'Star']]

    df = df[df.Star != 5]
    df = df[df.Star != 4]
    df = df[df.Star != 3]
    df.head()
    senti_list = []
    for i in df["Text"]:

        if (TextBlob(i).sentiment[0] > 0):
            senti_list.append('Positive')
        elif (TextBlob(i).sentiment[0] < 0):
            senti_list.append('Negative')
        else:
            senti_list.append('Neutral')

    df["sentiment"] = senti_list
    check_attention = df[(df["sentiment"] == "Positive") & (df["Star"] < 2)]

    st.dataframe(check_attention)
