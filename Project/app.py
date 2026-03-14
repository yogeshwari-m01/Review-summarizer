import streamlit as st
import pandas as pd
from collections import defaultdict
from transformers import pipeline
import re
from langdetect import detect
import matplotlib.pyplot as plt


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Hyderabad Business Review Analyzer",
    page_icon="📊",
    layout="wide"
)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("Project/hyderabad_reviews.csv")

df = load_data()


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline("sentiment-analysis")
    multi_sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    return sentiment_analyzer, multi_sentiment_analyzer


sentiment_analyzer, multi_sentiment_analyzer = load_models()


# ---------------- ASPECT KEYWORDS ----------------

aspects = {

'food_quality': [

# English
'biryani','food','taste','flavor','meal','coffee','tea','spicy','sweet','delicious','fresh',

# Hindi
'बिरयानी','खाना','स्वाद','चाय','कॉफी',

# Telugu
'బిర్యానీ','ఆహారం','రుచి','కాఫీ','టీ',

# Kannada
'ಬಿರಿಯಾನಿ','ಆಹಾರ','ರುಚಿ','ಕಾಫಿ','ಚಹಾ',

# Tamil
'பிரியாணி','உணவு','சுவை','காபி','தேநீர்',

# Roman Urdu
'biryani','khana','taste','chai','coffee'
],

'service': [

# English
'service','staff','wait','waiting','attentive','friendly','slow','rude','helpful',

# Hindi
'सेवा','स्टाफ','इंतजार',

# Telugu
'సర్వీస్','స్టాఫ్','వెయిట్',

# Kannada
'ಸೇವೆ','ಸಿಬ್ಬಂದಿ','ಕಾಯುವುದು',

# Tamil
'சேவை','பணியாளர்','காத்திருப்பு',

# Roman Urdu
'service','staff','wait','intezar'
],

'ambiance': [

# English
'ambiance','environment','atmosphere','decor','music','clean','comfortable','crowded',

# Hindi
'माहौल','साफ','संगीत',

# Telugu
'వాతావరణం','శుభ్రం','సంగీతం',

# Kannada
'ವಾತಾವರಣ','ಸ್ವಚ್ಛ','ಸಂಗೀತ',

# Tamil
'சூழல்','சுத்தம்','இசை',

# Roman Urdu
'ambiance','environment','saaf','music'
],

'price': [

# English
'price','cost','expensive','cheap','affordable','value','reasonable',

# Hindi
'कीमत','महंगा','सस्ता',

# Telugu
'ధర','ఖరీదు','చౌక',

# Kannada
'ಬೆಲೆ','ದುಬಾರಿ','ಸಸ್ತೆ',

# Tamil
'விலை','மிகவும் விலை','சுலபம்',

# Roman Urdu
'price','mehnga','sasta'
],

'shopping_experience': [

# English
'shopping','store','mall','brands','collection','parking','variety','available',

# Hindi
'खरीदारी','दुकान','मॉल',

# Telugu
'షాపింగ్','దుకాణం','మాల్',

# Kannada
'ಶಾಪಿಂಗ್','ದುಕಾಣ','ಮಾಲ್',

# Tamil
'ஷாப்பிங்','கடை','மால்',

# Roman Urdu
'shopping','store','mall'
]

}


# ---------------- LANGUAGE DETECTION ----------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"


# ---------------- SENTIMENT ANALYSIS ----------------

def analyze_sentiment_multilang(review):

    lang = detect_language(review)

    if lang == "en":
        result = sentiment_analyzer(review)[0]
        label = result["label"]
        score = result["score"]

    else:
        result = multi_sentiment_analyzer(review)[0]
        stars = int(result["label"][0])
        score = result["score"]

        if stars >= 4:
            label = "POSITIVE"
        elif stars == 3:
            label = "NEUTRAL"
        else:
            label = "NEGATIVE"

    return {"label": label, "score": score}


# ---------------- ASPECT EXTRACTION ----------------

def extract_aspect_sentiments_multilang(review):

    aspect_sentiments = {}

    for aspect, keywords in aspects.items():

        if any(re.search(r"\b" + kw + r"\b", review, re.I) for kw in keywords):

            sentiment = analyze_sentiment_multilang(review)

            aspect_sentiments[aspect] = sentiment

    return aspect_sentiments


# ---------------- PREPARE DATA ----------------

@st.cache_data
def prepare_data(df):

    df["aspect_sentiments"] = df["review_text"].apply(
        extract_aspect_sentiments_multilang
    )

    return df


df = prepare_data(df)


# ---------------- BUSINESS SUMMARY ----------------

def summarize_business(df, business):

    business_reviews = df[df["business_name"] == business]

    aspect_summary = defaultdict(lambda: {"positive":0,"negative":0})

    for aspects in business_reviews["aspect_sentiments"]:

        if not aspects:
            continue

        for aspect, sentiment in aspects.items():

            if sentiment["label"].lower() == "positive":
                aspect_summary[aspect]["positive"] += 1
            else:
                aspect_summary[aspect]["negative"] += 1


    summary = f"Summary for {business}:\n"

    for aspect, counts in aspect_summary.items():

        total = counts["positive"] + counts["negative"]

        if total == 0:
            continue

        pos_percent = (counts["positive"]/total)*100

        summary += f"- {aspect.capitalize()}: {pos_percent:.0f}% positive reviews\n"


    return summary, aspect_summary


# ---------------- DATAFRAME FOR CHART ----------------

def aspect_summary_df(aspect_summary):

    data = []

    for aspect, counts in aspect_summary.items():

        total = counts["positive"] + counts["negative"]

        if total == 0:
            continue

        pos_percent = (counts["positive"]/total)*100

        data.append({
            "Aspect": aspect.capitalize(),
            "Positive %": pos_percent
        })

    return pd.DataFrame(data)


# ---------------- SIDEBAR ----------------

st.sidebar.title("📊 Dashboard Menu")

menu = st.sidebar.radio(
    "Navigate",
    ["Business Analysis","Submit Review"]
)


# ================= BUSINESS ANALYSIS PAGE =================

if menu == "Business Analysis":

    st.title("📊 Hyderabad Business Review Analyzer")

    st.markdown(
    """
    Analyze customer reviews of **local Hyderabad businesses** using AI sentiment analysis.

    ✔ Aspect-based sentiment  
    ✔ Review insights  
    ✔ Business performance analysis
    """
    )


    businesses = df["business_name"].unique()


    st.subheader("Select Business")

    col1,col2 = st.columns([3,1])

    with col1:
        selected_business = st.selectbox(
            "Choose a business",
            businesses
        )

    with col2:
        analyze = st.button("Analyze")


    if analyze:

        summary_text, aspect_summary = summarize_business(df, selected_business)

        st.success(summary_text)


        business_reviews = df[df["business_name"] == selected_business]


        # -------- METRICS --------

        col1,col2,col3 = st.columns(3)

        total_reviews = len(business_reviews)

        positive_reviews = sum(
            1 for aspects in business_reviews["aspect_sentiments"]
            if aspects and any(v["label"]=="POSITIVE" for v in aspects.values())
        )

        negative_reviews = total_reviews - positive_reviews

        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive Reviews", positive_reviews)
        col3.metric("Negative Reviews", negative_reviews)


        # -------- BAR CHART --------

        st.subheader("Aspect Sentiment Analysis")

        df_plot = aspect_summary_df(aspect_summary)

        if not df_plot.empty:

            st.bar_chart(df_plot.set_index("Aspect"))


        # -------- PIE CHART --------

        labels = [k.capitalize() for k in aspect_summary.keys()]
        sizes = [v["positive"] for v in aspect_summary.values()]

        if sizes:

            fig, ax = plt.subplots()

            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%"
            )

            ax.set_title("Positive Aspect Distribution")

            st.pyplot(fig)


        # -------- REVIEWS --------

        st.subheader("Customer Reviews")

        for idx,row in business_reviews.iterrows():

            with st.expander(f"Review {idx+1}"):

                st.write(row["review_text"])

                aspect_data = row["aspect_sentiments"]

                if aspect_data:

                    for aspect,val in aspect_data.items():

                        st.write(
                            f"{aspect.capitalize()} → {val['label']} (Confidence {val['score']:.2f})"
                        )

                else:

                    st.write("No aspects detected")


# ================= SUBMIT REVIEW PAGE =================

if menu == "Submit Review":

    st.title("✍ Submit a New Review")


    new_business = st.text_input("Business Name")

    new_review = st.text_area("Write your review")


    submit = st.button("Submit Review")


    if submit:

        if new_business.strip()=="" or new_review.strip()=="":

            st.error("Please enter both business name and review.")

        else:

            new_aspect_sentiments = extract_aspect_sentiments_multilang(new_review)

            new_record = {
                "business_name": new_business.strip(),
                "review_text": new_review.strip(),
                "aspect_sentiments": new_aspect_sentiments
            }

            df.loc[len(df)] = new_record

            st.success("Review added successfully! Please go to Business Analysis to see results.")
