import streamlit as st
import pandas as pd
import pickle

from preprocessing import load_data, clean_data, select_features

st.set_page_config(page_title= "SPOTIFY WRAPPED", page_icon="🎧🎶", layout="wide")

@st.cache_resource    # cache prevents reloading everytime we want to interact with UI

def load_model():
    with open("model.pkl","rb") as f:
        model=pickle.load(f)

    with open("scaler.pkl","rb") as f:
        scaler=pickle.load(f)

    return model,scaler


@st.cache_data 

def load_and_prep_data():
    df=load_data()
    df=clean_data(df)
    X=select_features(df)

    return df, X

st.title("🎶🎶 Spotify Wrapped 🎶🎶")
st.write("Your personal music, powered by a simple ML model")

model,scaler=load_model()

df, X= load_and_prep_data()

X_scaled=scaler.transform(X)
df["predicted_mood"] = model.predict(X_scaled)

st.subheader("🎧Your Top Mood 🔥")
top_mood=df["predicted_mood"].value_counts().idxmax()

st.success(f"Your most common VIBE 👌 is.....🥁🥁\n **{top_mood}**")

st.subheader("📊 Mood Distribution")
st.bar_chart(df["predicted_mood"].value_counts())

st.subheader("🎧 Your Audio Feature Report 📃")
feature_means = X.mean()
st.bar_chart(feature_means)

st.subheader("🔥 NOW FOR THE FUN INSIGHTS!!!! 🔥")

if top_mood == "Energetic":
    st.write("You love high-energy tracks — probably gym or hype playlists 💪")
elif top_mood == "Chill":
    st.write("You prefer calm, acoustic vibes — late night listener 🌙")
elif top_mood == "Happy":
    st.write("You enjoy upbeat, feel-good music — main character energy ✨")
else:
    st.write("You lean towards emotional tracks — deep feels 🎭")

