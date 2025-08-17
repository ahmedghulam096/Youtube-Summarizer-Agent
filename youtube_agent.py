# app.py
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
import os
from dotenv import load_dotenv
import yt_dlp
import re
import requests
import traceback
import time
from datetime import datetime
import math

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# -------- Helper functions --------
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|/embed/|/v/|shorts/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError("Couldn't extract a valid YouTube video ID.")
    return match.group(1)

def format_date(date_str):
    if not date_str:
        return "Unknown Date"
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%b %d, %Y")
    except:
        return date_str

def fetch_video_details(video_url: str):
    ydl_opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return {
        "title": info.get("title", "Unknown Title"),
        "thumbnail": info.get("thumbnail", ""),
        "uploader": info.get("uploader", "Unknown Uploader"),
        "upload_date": info.get("upload_date", ""),
        "view_count": info.get("view_count", 0),
        "duration": info.get("duration", 0),
        "description": info.get("description", "")
    }

def fetch_transcript_yt_dlp(video_url: str) -> str:
    try:
        ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        subs = info.get("automatic_captions") or info.get("subtitles") or {}
        if not subs:
            return None, None
        lang_key = "en" if "en" in subs else next(iter(subs.keys()))
        entry = subs[lang_key][0]
        sub_url = entry.get("url")
        if not sub_url:
            return None, None
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(sub_url, headers=headers, timeout=15)
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, lang_key
    except Exception:
        traceback.print_exc()
        return None, None

def chunk_text(text: str, max_words: int = 200):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s_len = len(s.split())
        if cur_len + s_len > max_words and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], s_len
        else:
            cur.append(s)
            cur_len += s_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# -------- AI Agent --------
youtube_agent = Agent(
    name="YouTube Summarizer",
    role="Summarize YouTube video transcript",
    model=Groq(id="compound-beta-mini"),
    instructions="Produce a concise and clear summary based purely on the transcript text.",
    show_tool_calls=False,
    markdown=True,
)

def summarize_using_agent(chunks: list, style="concise"):
    summaries = []
    delay_seconds = 0.7

    for i, chunk in enumerate(chunks, start=1):
        st.progress(i / len(chunks), text=f"Summarizing chunk {i}/{len(chunks)}...")
        if style == "detailed":
            prompt = f"Summarize chunk {i} in detail:\n{chunk}"
        else:
            prompt = f"Summarize chunk {i} into concise bullet points:\n{chunk}"

        out = youtube_agent.run(prompt)
        summaries.append(out.strip() if isinstance(out, str) else str(out))
        time.sleep(delay_seconds)

    final_prompt = "Combine the following summaries into a polished final summary:\n\n" + "\n\n".join(summaries)
    final_summary = youtube_agent.run(final_prompt)
    return final_summary.strip() if isinstance(final_summary, str) else str(final_summary)

# -------- Streamlit UI --------
st.set_page_config(page_title="🎬 AI Video Summarizer", page_icon="🎬", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; background: -webkit-linear-gradient(#ff7e5f, #feb47b); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    🎬 AI Video Summarizer
    </h1>
    """, 
    unsafe_allow_html=True
)
st.caption("Transform any YouTube video into professional summaries in seconds 🚀")

# Input
video_url = st.text_input("🎥 Paste YouTube URL (normal or Shorts):", placeholder="https://www.youtube.com/watch?v=...")

summary_style = st.radio("📑 Choose summary style:", ["Concise", "Detailed"], horizontal=True)

if st.button("⚡ Summarize Video"):
    if not video_url.strip():
        st.warning("Please paste a valid YouTube video URL.")
    else:
        try:
            start_time = time.time()
            video_id = extract_video_id(video_url)
            details = fetch_video_details(video_url)

            # --- Video Info Section ---
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    if details["thumbnail"]:
                        st.image(details["thumbnail"], use_container_width=True)
                with col2:
                    st.markdown(f"### {details['title']}")
                    st.markdown(f"📺 **Uploader:** {details['uploader']}")
                    st.markdown(f"📅 **Upload Date:** {format_date(details['upload_date'])}")
                    st.markdown(f"👁 **Views:** {details['view_count']:,}")
                    minutes, seconds = divmod(details['duration'], 60)
                    st.markdown(f"⏱ **Duration:** {minutes}m {seconds}s")
            with st.expander("📄 Video Description", expanded=False):
                st.write(details["description"])

            st.divider()

            # Fetch transcript
            transcript, lang = fetch_transcript_yt_dlp(video_url)
            if not transcript:
                st.error("❌ Transcript unavailable; cannot summarize.")
                st.stop()

            # Transcript info only (no full transcript shown)
            word_count = len(transcript.split())
            read_time = math.ceil(word_count / 200)  # avg 200 wpm
            st.info(f"📝 Transcript length: {word_count} words | ⏱ Estimated reading time: {read_time} min | 🌐 Language: {lang}")

            # Split and summarize
            chunks = chunk_text(transcript, max_words=200)
            st.write(f"✂ Transcript split into **{len(chunks)} chunks**.")

            with st.spinner("🧠 Summarizing with AI..."):
                summary = summarize_using_agent(chunks, style=summary_style.lower())

            st.success("✅ Summary Generated")
            st.subheader("📜 Final Summary")
            st.markdown(summary)

            # Downloads
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download as TXT",
                    data=summary,
                    file_name=f"{details['title'][:50]}_AI_Summary.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="📄 Download as Markdown",
                    data=summary,
                    file_name=f"{details['title'][:50]}_AI_Summary.md",
                    mime="text/markdown"
                )

            st.caption(f"⏱ Process completed in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            st.error("⚠️ Something went wrong while processing the video.")
            st.exception(e)
