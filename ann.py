import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt

#lets just get the dataset
df=pd.read_csv("Reviews(1).csv") 
df = df.dropna(subset=['Text'])
review_text = df['Text'].astype(str)
 
#classify
analyzer=SentimentIntensityAnalyzer()

sentiment_scores=[]
blob_subj=[]
for review in review_text:
        sentiment_scores.append(analyzer.polarity_scores(review)["compound"])
        blob=TextBlob(review)
        blob_subj.append(blob.subjectivity)

sentiment_classes=[]
for sentiment_score in sentiment_scores:
    if sentiment_score > 0.8:
        sentiment_classes.append("higly positive")
    elif sentiment_score > 0.4:
        sentiment_classes.append("positive")
    elif -0.4 <= sentiment_score < 0.4:
        sentiment_classes.append("neutral")
    elif sentiment_score < -0.4:
        sentiment_classes.append("negative")
    else:
        sentiment_classes.append("higly negative")

#streamlit
st.title("Sentiment Analysis On Customer Feedback")

#user input
user_input = st.text_area("Enter the feedback")
blob=TextBlob(user_input)
 
user_sentiment_score = analyzer.polarity_scores(user_input)['compound']
if user_sentiment_score > 0.8:
    user_sentiment_class = "higly positive"
elif user_sentiment_score > 0.4:
    user_sentiment_class = "positive"
elif -0.4 <= user_sentiment_score <= 0.4:
    user_sentiment_class = "neutral"
elif user_sentiment_score < -0.4:
    user_sentiment_class = "negative"
else:
    user_sentiment_class = "higly negative"

st.write("** VADER Sentiment Class: **", user_sentiment_class, "**Vader Sentiment Scores: **",user_sentiment_score)
st.write("** TextBlob Polarity **",blob.sentiment.polarity,"**TextBlob subjectivity: **", blob.sentiment.subjectivity)

#Display clean text
pre=st.text_input('clean Text:')
if pre:
    st.write(cleantext.clean(pre, clean_all= False, extra_spaces= True, stopwords= True, lowercase= True, numbers= True, punct= True))
else:
    st.write("No Text is been provided from the user for cleaning")

#graphical representation of the data
st.subheader("Graphical representation of Data")
plt.figure(figsize=(10,6))

sentiment_scores_by_class= {k: [] for k in set(sentiment_classes)}
for sentiment_score, sentiment_class in zip(sentiment_scores, sentiment_classes):
    sentiment_scores_by_class[sentiment_class].append(sentiment_score)

for sentiment_class, scores in sentiment_scores_by_class.items():
    plt.hist(scores, label=sentiment_class, alpha=0.5)

plt.xlabel("sentiment score")
plt.ylabel("Count")
plt.title("Score distribution by class")
plt.legend()
st.pyplot(plt)

#DataFrameswith the sentiment analysis results

df["Sentiment Class"]= sentiment_classes
df["Sentiment Score"]= sentiment_scores
df["Subject"]= blob_subj

new_df = df[["Score","Text","Sentiment score","Sentiment class","Subjectivity"]]
st.subheader("Input Dataframe")
st.dataframe(new_df.head(10),use_container_width=True)