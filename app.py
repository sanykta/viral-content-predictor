import numpy as np
import pandas as pd
import joblib
import re
import streamlit as st

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from transformers.pipelines import pipeline

@st.cache_resource
def setup_nltk():

    nltk.download(
        "vader_lexicon"
    )


setup_nltk()

sia = SentimentIntensityAnalyzer()

@st.cache_resource
def load_models():

    viral_model = joblib.load(
        "viral_predictor_pipeline (1).pkl"
    )

    threshold = joblib.load(
        "threshold (1).pkl"
    )

    performance_model = joblib.load(
        "performance_category_pipeline (1).pkl"
    )

    label_encoder = joblib.load(
        "category_label_encoder (1).pkl"
    )

    return (
        viral_model,
        threshold,
        performance_model,
        label_encoder
    )


(
    viral_model,
    threshold,
    performance_model,
    label_encoder
) = load_models()


@st.cache_resource
def load_emotion_model():

    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None
    )

def get_sentiment(text):

    sentiment_score = sia.polarity_scores(
        text
    )["compound"]


    if sentiment_score >= 0.05:

        sentiment = "positive"


    elif sentiment_score <= -0.05:

        sentiment = "negative"


    else:

        sentiment = "neutral"


    return (
        sentiment,
        sentiment_score
    )

def get_emotions(text):
    
    emotion_model = load_emotion_model()

    result = emotion_model(
        text,
        truncation=True
    )[0]


    result = sorted(
        result,
        key=lambda x: x["score"],
        reverse=True
    )


    emotion = result[0]["label"]

    emotion_score = result[0]["score"]


    emotional_trigger = result[1]["label"]

    emotional_trigger_score = result[1]["score"]


    return (
        emotion,
        emotion_score,
        emotional_trigger,
        emotional_trigger_score
    )

def detect_content_type(text):

    text = text.lower()


    if any(word in text for word in ["top", "best", "worst", "ranking", "ranked"]):

        return "ranking"


    elif any(word in text for word in ["explained", "ending", "theory", "breakdown"]):

        return "explained"


    elif any(word in text for word in ["review", "honest opinion", "worth watching"]):

        return "review"


    elif any(word in text for word in ["reaction", "reacts", "reacting"]):

        return "reaction"


    elif any(word in text for word in ["news", "announced", "release", "update"]):

        return "news"


    else:

        return "other"

from datetime import datetime

def prepare_input(
    title,
    description,
    duration_seconds,
    subscriber_count,
    channel_avg_views,
    upload_day,
    upload_hour
):


    text = title + " " + description #combining text

    # Title Features
    

    title_length = len(title)

    title_word_count = len(
        title.split()
    )


    has_number = int(
        bool(
            re.search(
                r"\d",
                title
            )
        )
    )


    has_question = int(
        "?" in title
    )


    has_exclamation = int(
        "!" in title
    )


    uppercase_ratio = (
        sum(
            1 for c in title
            if c.isupper()
        )
        /
        max(len(title), 1)
    )


  
    # Hashtag Features
    

    hashtags = re.findall(
        r"#\w+",
        text
    )


    hashtag_count = len(
        hashtags
    )


    has_hashtag = int(
        hashtag_count > 0
    )


    # Clean Text
    

    clean_text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text.lower()
    )


   
    # Date / Time Features
 

    upload_month = datetime.now().month


    is_weekend = int(
        upload_day in [
            "Saturday",
            "Sunday"
        ]
    )

    # VADER Sentiment
   

    sentiment, sentiment_score = get_sentiment(
        text
    )
   
    # RoBERTa Emotion


    (
        emotion,
        emotion_score,
        emotional_trigger,
        emotional_trigger_score

    ) = get_emotions(
        text
    )

    # Content Type

    content_type = detect_content_type(
        text
    )
   
    # Channel Features

    views_per_subscriber = (
        channel_avg_views /
        (subscriber_count + 1)
    )

    # Final Input DataFrame
   
    input_df = pd.DataFrame(
        {

            # adding numeric features

            "title_length": [title_length],

            "title_word_count": [title_word_count],

            "has_number": [has_number],

            "has_question": [has_question],

            "has_exclamation": [has_exclamation],

            "uppercase_ratio": [uppercase_ratio],

            "upload_month": [upload_month],

            "upload_hour": [upload_hour],

            "is_weekend": [is_weekend],

            "duration_seconds": [duration_seconds],

            "hashtag_count": [hashtag_count],

            "has_hashtag": [has_hashtag],

            "sentiment_score": [sentiment_score],

            "emotion_score": [emotion_score],

            "emotional_trigger_score": [emotional_trigger_score],

            "views_per_subscriber": [views_per_subscriber],

            "subscriber_count": [subscriber_count],

            "channel_avg_views": [channel_avg_views],


            # adding categorical features

            "sentiment": [sentiment],

            "emotion": [emotion],

            "emotional_trigger": [emotional_trigger],

            "content_type": [content_type],

            "upload_day": [upload_day],


            # TF-IDF text

            "clean_text": [clean_text]

        }
    )


    return input_df

def get_creator_score(probability):

    creator_score = np.interp(
        probability,
        [0, threshold, 1],
        [0, 60, 100]
    )


    return round(
        creator_score,
        2
    )

def generate_recommendations(input_data):

    recommendations = []

    # Title length

    title_words = input_data[
        "title_word_count"
    ].iloc[0]


    if title_words < 5:

        recommendations.append(
            "Try a more descriptive title with stronger keywords."
        )


    if title_words > 15:

        recommendations.append(
            "Consider shortening the title for better readability."
        )

    # Numbers/List format

    has_number = input_data[
        "has_number"
    ].iloc[0]


    if has_number == 0:

        recommendations.append(
            "List-style titles with numbers often create stronger curiosity hooks."
        )

    # Question hook

    has_question = input_data[
        "has_question"
    ].iloc[0]


    if has_question == 0:

        recommendations.append(
            "Try adding curiosity-driven questions when relevant."
        )

    # Hashtags

    hashtag_count = input_data[
        "hashtag_count"
    ].iloc[0]


    if hashtag_count == 0:

        recommendations.append(
            "Add relevant hashtags to improve discoverability."
        )

    # Emotional hook

    emotion = input_data[
        "emotion"
    ].iloc[0]


    if emotion == "neutral":

        recommendations.append(
            "Add stronger emotional triggers like surprise, curiosity, or excitement."
        )

    # Content format

    content_type = input_data[
        "content_type"
    ].iloc[0]


    if content_type == "other":

        recommendations.append(
            "Try clearer formats like rankings, reviews, reactions, or explanations."
        )

    # Upload timing

    upload_hour = input_data[
        "upload_hour"
    ].iloc[0]


    if upload_hour < 12:

        recommendations.append(
            "Consider testing afternoon/evening upload timings."
        )

    if len(recommendations) == 0:

        recommendations.append(
            "Strong content structure detected. Focus on thumbnail and audience retention."
        )

    return recommendations

def predict_content(
    title,
    description,
    duration_seconds,
    subscriber_count,
    channel_avg_views,
    upload_day,
    upload_hour
):

    # Creating features

    input_data = prepare_input(

        title=title,

        description=description,

        duration_seconds=duration_seconds,

        subscriber_count=subscriber_count,

        channel_avg_views=channel_avg_views,

        upload_day=upload_day,

        upload_hour=upload_hour
    )

    #Viral probability

    viral_probability = viral_model.predict_proba(
        input_data
    )[0][1]


    creator_score = get_creator_score(
        viral_probability
    )


    if viral_probability >= threshold:

        viral_prediction = "🔥 Viral Potential"

    else:

        viral_prediction = "Needs Optimization"

    # Performance category
    category_prediction = performance_model.predict(
        input_data
    )
    
    performance_category = (
        label_encoder.inverse_transform(
            category_prediction
        )[0]
    )

    # Recommendations

    recommendations = generate_recommendations(
        input_data
    )

    # Returning final user output

    return {

        "Creator Score": f"{creator_score}%",

        "Viral Prediction": viral_prediction,

        "Expected Performance": performance_category,

        "Recommendations": recommendations
    }

# Streamlit UI

st.title(
    "🚀 Viral Content Predictor"
)


st.write(
    "Predict your video's viral potential using AI"
)


title = st.text_input(
    "Video Title"
)


description = st.text_area(
    "Video Description"
)


duration_seconds = st.number_input(
    "Video Duration (seconds)",
    min_value=0,
    value=300
)


subscriber_count = st.number_input(
    "Subscriber Count",
    min_value=0,
    value=1000
)


channel_avg_views = st.number_input(
    "Channel Average Views",
    min_value=0,
    value=1000
)


upload_day = st.selectbox(
    "Upload Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)


upload_hour = st.slider(
    "Upload Hour (24H format)",
    0,
    23,
    18
)


if st.button(
    "Predict Viral Potential"
):

    result = predict_content(

        title,
        description,
        duration_seconds,
        subscriber_count,
        channel_avg_views,
        upload_day,
        upload_hour
    )


    st.metric(
        "Creator Score",
        result["Creator Score"]
    )


    st.success(
        result["Viral Prediction"]
    )


    st.write(
        "Expected Performance:",
        result["Expected Performance"]
    )


    st.subheader(
        "Recommendations"
    )


    for r in result["Recommendations"]:

        st.write(
            "•",
            r
        )
