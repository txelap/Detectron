import streamlit as st
import json
from datetime import datetime
from pipeline import DisinformationPipeline

# Configure Streamlit page
st.set_page_config(page_title="Disinformation Detector", page_icon="🛡️", layout="wide")

# Initialize Pipeline (cached so it only loads models once)
@st.cache_resource
def load_pipeline():
    return DisinformationPipeline()

st.title("🛡️ Real-Time Disinformation Detector")
st.markdown("Analyze social media posts and news articles across 7 different axes using Artificial Intelligence.")

pipeline = load_pipeline()

# Input UI
st.sidebar.header("Input Data")
st.sidebar.markdown("Provide the details of the post or article you want to verify.")

headline = st.sidebar.text_input("Headline / Title", "Nuevas medidas económicas anunciadas")
content = st.sidebar.text_area("Content / Body", "El ministerio de economía ha publicado hoy el nuevo conjunto de medidas.")
source_url = st.sidebar.text_input("Source URL", "https://www.reuters.com/news/123")
media_url = st.sidebar.text_input("Image/Media URL", "")
reported_date = st.sidebar.date_input("Reported Date", datetime.now())

st.sidebar.markdown("---")
st.sidebar.subheader("User Profile (Social Media)")
followers = st.sidebar.number_input("Followers", value=1000)
following = st.sidebar.number_input("Following", value=100)
has_pic = st.sidebar.checkbox("Has Profile Picture", value=True)

if st.sidebar.button("Analyze Content", type="primary"):
    with st.spinner("Running AI analysis across 7 modules..."):

        # Format data for pipeline
        post_data = {
            "headline": headline,
            "content": content,
            "source_url": source_url,
            "media_url": media_url,
            "reported_date": reported_date.strftime("%Y-%m-%d"),
            "user_profile": {
                "created_at": "2020-01-01",
                "followers_count": followers,
                "following_count": following,
                "has_profile_pic": has_pic,
                "statuses_count": 500
            }
        }

        # Run Analysis
        report = pipeline.analyze_post(post_data)

        # Display Results
        st.header("Analysis Results")

        # Main Score
        final_score = report["final_score"]["disinformation_probability"]
        assessment = report["final_score"]["assessment"]

        if final_score > 0.7:
            color = "red"
        elif final_score > 0.4:
            color = "orange"
        else:
            color = "green"

        st.markdown(f"### Disinformation Risk: <span style='color:{color}'>{final_score*100:.1f}%</span>", unsafe_allow_html=True)
        st.subheader(f"Assessment: {assessment}")
        st.progress(final_score)

        st.markdown("---")
        st.markdown("### The 7 Axes of Analysis")

        col1, col2 = st.columns(2)

        analysis = report["analysis"]

        with col1:
            st.markdown("#### Text & Sentiment")
            st.write(f"**Clickbait Score:** {analysis['2_clickbait_analysis']['score']:.2f}")
            st.info(analysis['2_clickbait_analysis']['details'])

            st.write(f"**Emotion/Manipulation:** {analysis['3_emotion_analysis']['score']:.2f}")
            st.info(analysis['3_emotion_analysis']['details'])

            st.write(f"**Evidence Consensus:** {analysis['5_evidence_search']['score']:.2f}")
            st.info(analysis['5_evidence_search']['details'])

        with col2:
            st.markdown("#### Metadata & Media")
            st.write(f"**Source Unreliability:** {1.0 - analysis['1_source_reliability']['score']:.2f}")
            st.info(analysis['1_source_reliability']['details'])

            st.write(f"**Recycled News (Date):** {analysis['4_date_verification']['score']:.2f}")
            st.info(analysis['4_date_verification']['details'])

            st.write(f"**Bot Profile:** {analysis['6_profile_analysis']['score']:.2f}")
            st.info(analysis['6_profile_analysis']['details'])

            st.write(f"**AI/Deepfake Media:** {analysis['7_media_analysis']['score']:.2f}")
            st.info(analysis['7_media_analysis']['details'])

        with st.expander("View Raw JSON Report"):
            st.json(report)
