import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="CLAT Mentor Recommender", layout="centered")

# Load mentors
def load_mentors():
    if os.path.exists("mentors.csv"):
        return pd.read_csv("mentors.csv")
    else:
        # Create a sample template if not found
        df = pd.DataFrame(columns=[
            "mentor_id", "name", "preferred_subjects", "target_colleges",
            "prep_level", "learning_style"
        ])
        df.to_csv("mentors.csv", index=False)
        return df

# Save new mentor
def save_mentor(data):
    df = load_mentors()
    data['mentor_id'] = len(df) + 1
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv("mentors.csv", index=False)

# Recommendation logic
def recommend_mentors(aspirant_profile, mentors_df):
    aspirant_combined = aspirant_profile['preferred_subjects'] + ' ' + \
                        aspirant_profile['target_colleges'] + ' ' + \
                        aspirant_profile['prep_level'] + ' ' + \
                        aspirant_profile['learning_style']

    mentors_df['combined_features'] = mentors_df['preferred_subjects'] + ' ' + \
                                      mentors_df['target_colleges'] + ' ' + \
                                      mentors_df['prep_level'] + ' ' + \
                                      mentors_df['learning_style']

    vectorizer = TfidfVectorizer()
    mentor_vectors = vectorizer.fit_transform(mentors_df['combined_features'])
    aspirant_vector = vectorizer.transform([aspirant_combined])

    similarity_scores = cosine_similarity(aspirant_vector, mentor_vectors).flatten()
    mentors_df['similarity_score'] = similarity_scores

    return mentors_df.sort_values(by='similarity_score', ascending=False).head(3)

# Navigation
page = st.sidebar.selectbox("Choose View", ["🔍 Aspirant - Get Recommendations", "🧑‍🏫 Mentor - Register Yourself"])

# ==========================
# 🔍 Aspirant View
# ==========================
if page.startswith("🔍"):
    st.title("🎯 CLAT Mentor Recommendation")

    mentors = load_mentors()

    if len(mentors) == 0:
        st.warning("No mentors in the database yet. Please add mentors.")
    else:
        st.sidebar.header("Aspirant Preferences")

        preferred_subjects = st.sidebar.selectbox("Preferred Subject", mentors['preferred_subjects'].dropna().unique())
        target_colleges = st.sidebar.selectbox("Target College", mentors['target_colleges'].dropna().unique())
        prep_level = st.sidebar.selectbox("Preparation Level", ['Beginner', 'Intermediate', 'Advanced'])
        learning_style = st.sidebar.selectbox("Learning Style", mentors['learning_style'].dropna().unique())

        aspirant_profile = {
            'preferred_subjects': preferred_subjects,
            'target_colleges': target_colleges,
            'prep_level': prep_level,
            'learning_style': learning_style
        }

        if st.sidebar.button("🚀 Recommend Mentors"):
            top_matches = recommend_mentors(aspirant_profile, mentors.copy())

            st.subheader("✅ Top 3 Mentor Matches")
            st.dataframe(top_matches[['mentor_id', 'name', 'similarity_score']].reset_index(drop=True))

            st.subheader("📊 Visual Match Scores")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='similarity_score', y='name', data=top_matches, palette='viridis', ax=ax)
            ax.set_title('Top 3 Mentor Recommendations')
            st.pyplot(fig)

# ==========================
# 🧑‍🏫 Mentor Registration View
# ==========================
else:
    st.title("🧑‍🏫 Mentor Registration Form")

    with st.form("mentor_form"):
        name = st.text_input("Full Name")
        preferred_subjects = st.text_input("Preferred Subjects (e.g., Legal Reasoning, GK)")
        target_colleges = st.text_input("Target Colleges (e.g., NLSIU, NALSAR)")
        prep_level = st.selectbox("Prep Level to Guide", ['Beginner', 'Intermediate', 'Advanced'])
        learning_style = st.selectbox("Preferred Learning Style", ['Visual', 'Auditory', 'Kinesthetic', 'Reading/Writing'])

        submitted = st.form_submit_button("Submit")

        if submitted:
            if not name or not preferred_subjects or not target_colleges:
                st.error("Please fill all the required fields.")
            else:
                save_mentor({
                    'name': name,
                    'preferred_subjects': preferred_subjects,
                    'target_colleges': target_colleges,
                    'prep_level': prep_level,
                    'learning_style': learning_style
                })
                st.success(f"Thank you, {name}! You’ve been added as a mentor. 🎉")
