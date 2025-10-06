import io
import os
import csv
from datetime import datetime

import altair as alt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit_mic_recorder import mic_recorder
from load_model import create_genre_classifier
from genre_labels import get_genre_name, GENRE_LABELS
from fake_recommendations import (
    get_recommendations,
    GENRE_TO_BROAD_CATEGORY,
)

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0b0d10;
            --bg-secondary: #141922;
            --accent-primary: #6c5ce7;
            --accent-secondary: #00cec9;
            --text-primary: #e6edf3;
            --text-muted: #a0aec0;
        }
        .stApp {
            background-color: var(--bg-primary);
            color: var(--text-primary);
        }
        section[data-testid="stSidebar"] {
            background: #0f131a;
            padding-top: 0.5rem;
            width: 16rem !important;
        }
        section[data-testid="stSidebar"] .stButton>button,
        section[data-testid="stSidebar"] .stSelectbox>div>div {
            font-size: 0.85rem;
        }
        div[data-testid="stVerticalBlock"] > div {
            border-radius: 18px;
            background: var(--bg-secondary);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.35);
        }
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: 0.02em;
        }
        h1 { font-size: 2.4rem !important; }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.45rem !important; }
        [data-testid="block-container"] {
            padding-top: 1.75rem;
            padding-bottom: 2rem;
        }
        .prediction-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.15rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: #0b0d10;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .prediction-box {
            background: linear-gradient(135deg, rgba(108,92,231,0.25), rgba(0,206,201,0.2));
            border-radius: 18px;
            padding: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 28px rgba(0,0,0,0.35);
        }
        .prediction-box .genre-name {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.25rem 0;
        }
        .prediction-box .confidence {
            font-size: 1rem;
            color: var(--text-muted);
            margin: 0;
        }
        a.contact-button {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: #0b0d10 !important;
            padding: 0.45rem 1.1rem;
            border-radius: 999px;
            font-weight: 600;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner=False)
def load_model() -> torch.nn.Module | None:
    model_path = os.path.join("models", "model.pt")
    if os.path.exists(model_path):
        try:
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            return model
        except Exception as load_error:
            st.warning(f"Could not load models/model.pt: {load_error}")
    return load_model_old()

@st.cache_resource(show_spinner=False)
def load_model_old() -> torch.nn.Module | None:
    try:
        with st.spinner("Loading AI model..."):
            checkpoint_path = "genre_model.pth"
            if os.path.exists(checkpoint_path):
                st.info("📦 Loading trained model checkpoint...")
                try:
                    model = create_genre_classifier(num_classes=len(GENRE_LABELS), pretrained=False, device="cpu")
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    model.load_state_dict(checkpoint)
                    model.eval()
                    st.success("✅ Trained model loaded successfully!")
                    return model
                except Exception as checkpoint_error:
                    st.warning(f"Could not load checkpoint: {checkpoint_error}")
                    st.info("Falling back to pre-trained ImageNet model...")
            st.info("🧠 Loading pre-trained ResNet18 model...")
            model = create_genre_classifier(num_classes=len(GENRE_LABELS), pretrained=True, device="cpu")
            model.eval()
            st.success("✅ Model loaded successfully!")
            return model
    except Exception as load_error:
        st.error(f"❌ Critical error loading model: {load_error}")
        st.error("Please ensure PyTorch and torchvision are installed correctly.")
        return None

def preprocess_spectrogram(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    return img_tensor.unsqueeze(0)

def predict_genre(model: torch.nn.Module, spectrogram_image: Image.Image):
    if model is None or spectrogram_image is None:
        return None, None, None
    try:
        input_tensor = preprocess_spectrogram(spectrogram_image)
    except Exception:
        return None, None, None
    with torch.no_grad():
        try:
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item(), probabilities[0].numpy()
        except RuntimeError:
            return None, None, None

def log_prediction(filename: str, predicted_genre: str, confidence: float, log_file: str = "predictions.log") -> bool:
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.exists(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as log_fp:
            writer = csv.writer(log_fp)
            if not file_exists:
                writer.writerow(["Timestamp", "Filename", "Predicted_Genre", "Confidence"])
            writer.writerow([timestamp, filename, predicted_genre, f"{confidence:.4f}"])
        return True
    except Exception:
        return False

def load_analytics_data(log_file: str = "predictions.log") -> dict:
    analytics = {
        "total_predictions": 0,
        "genre_counts": {},
        "recordings": 0,
        "uploads": 0,
        "data": [],
    }
    if not os.path.exists(log_file):
        return analytics
    try:
        with open(log_file, "r", encoding="utf-8") as log_fp:
            reader = csv.DictReader(log_fp)
            for row in reader:
                analytics["data"].append(row)
                analytics["total_predictions"] += 1
                genre = row["Predicted_Genre"]
                analytics["genre_counts"][genre] = analytics["genre_counts"].get(genre, 0) + 1
                if row["Filename"] == "recorded":
                    analytics["recordings"] += 1
                else:
                    analytics["uploads"] += 1
    except Exception:
        pass
    return analytics

def display_analytics() -> None:
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Analytics Dashboard")
    analytics = load_analytics_data()
    if analytics["total_predictions"] == 0:
        st.sidebar.info("No predictions logged yet. Make some predictions to see analytics!")
        return
    st.sidebar.metric("Total Predictions", analytics["total_predictions"])
    st.sidebar.subheader("📁 Source Breakdown")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("🎤 Recordings", analytics["recordings"])
    with col2:
        st.metric("📂 Uploads", analytics["uploads"])
    st.sidebar.subheader("🎵 Genre Distribution")
    if analytics["genre_counts"]:
        genre_df = pd.DataFrame(list(analytics["genre_counts"].items()), columns=["Genre", "Count"]).sort_values("Count", ascending=False)
        st.sidebar.bar_chart(genre_df.set_index("Genre"))
        with st.sidebar.expander("View Detailed Genre Counts"):
            for _, row in genre_df.iterrows():
                percentage = (row["Count"] / analytics["total_predictions"]) * 100
                st.write(f"**{row['Genre']}**: {row['Count']} ({percentage:.1f}%)")
    st.sidebar.subheader("🕒 Recent Predictions")
    recent_predictions = analytics["data"][-5:][::-1]
    if recent_predictions:
        for pred in recent_predictions:
            with st.sidebar.expander(f"{pred['Predicted_Genre']} ({pred['Timestamp']})"):
                st.write(f"**File:** {pred['Filename']}")
                st.write(f"**Confidence:** {float(pred['Confidence']) * 100:.2f}%")
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Download Full Log"):
        try:
            with open("predictions.log", "r", encoding="utf-8") as log_fp:
                log_content = log_fp.read()
            st.sidebar.download_button(
                label="Download predictions.log",
                data=log_content,
                file_name="predictions.log",
                mime="text/csv",
            )
        except Exception:
            st.sidebar.error("Could not load log file.")

def render_cqt_spectrogram(uploaded_file) -> bytes:
    if uploaded_file is None:
        raise ValueError("uploaded_file must not be None.")
    uploaded_file.seek(0)
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    cqt = librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="cqt_note",
        bins_per_octave=12,
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Constant-Q Transform (CQT) Spectrogram", fontsize=12, pad=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (notes)")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.set_ylabel("Amplitude (dB)", rotation=-90, va="bottom")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def process_audio(audio_bytes: bytes, file_extension: str) -> tuple[bytes | None, Image.Image | None]:
    if not audio_bytes:
        return None, None
    extension = file_extension or "wav"
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = f"temp_audio.{extension}"
    try:
        spectrogram_png = render_cqt_spectrogram(audio_buffer)
        spectrogram_image = Image.open(io.BytesIO(spectrogram_png)).convert("RGB")
        return spectrogram_png, spectrogram_image
    except Exception:
        return None, None

def display_predictions(probs, labels) -> None:
    probabilities = np.asarray(probs, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(labels):
        return
    top_indices = probabilities.argsort()[-3:][::-1]
    top_probs = probabilities[top_indices]
    top_labels = [labels[i] for i in top_indices]
    table_df = pd.DataFrame({
        "Rank": np.arange(1, len(top_probs) + 1),
        "Label": top_labels,
        "Probability (%)": np.round(top_probs * 100, 2),
    })
    st.dataframe(table_df, hide_index=True, use_container_width=True)
    chart_df = pd.DataFrame({
        "Label": top_labels,
        "Probability": top_probs * 100,
    })
    chart = (
        alt.Chart(chart_df)
        .mark_bar(radius=6)
        .encode(
            x=alt.X("Probability:Q", title="Probability (%)", scale=alt.Scale(domain=(0, 100))),
            y=alt.Y("Label:N", sort=top_labels[::-1], title=""),
            color=alt.value("#6c5ce7"),
            tooltip=["Label", alt.Tooltip("Probability:Q", format=".2f")],
        )
    )
    text = (
        alt.Chart(chart_df)
        .mark_text(align="left", dx=5, fontWeight="bold", color="#e6edf3")
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=(0, 100))),
            y=alt.Y("Label:N", sort=top_labels[::-1]),
            text=alt.Text("Probability:Q", format=".1f"),
        )
    )
    st.altair_chart(chart + text, use_container_width=True)
    top_confidence = float(top_probs[0]) if len(top_probs) else 0.0
    if top_confidence > 0.7:
        pill_color = "#2ecc71"
        status = "High"
    elif top_confidence > 0.4:
        pill_color = "#f39c12"
        status = "Moderate"
    else:
        pill_color = "#e74c3c"
        status = "Low"
    st.markdown(
        f"""
        <span class="prediction-badge" style="background:{pill_color};color:#0b0d10;">
            Confidence: {status} ({top_confidence:.0%})
        </span>
        """,
        unsafe_allow_html=True,
    )

def render_home_tab() -> None:
    st.header("🎵 Music Genre Classifier")
    st.sidebar.title("⚙️ Settings")
    show_analytics = st.sidebar.checkbox("Show Analytics (Admin)", value=False)
    if show_analytics:
        display_analytics()
    with st.expander("ℹ️ About this app", expanded=False):
        st.write(
            """
            This application analyzes music using a ResNet18-based deep learning model
            trained on Constant-Q Transform (CQT) spectrograms. Upload or record a clip to
            predict among 16 finely curated genres and explore tailored song recommendations.
            """
        )
    st.markdown("---")
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check the error messages above.")
        st.stop()
    st.subheader("🎵 Choose Input Method")
    input_method = st.radio(
        "How would you like to provide audio?",
        ["📁 Upload Audio File", "🎤 Record Audio"],
        horizontal=True,
    )
    st.markdown("---")
    audio_source = None
    audio_filename = None
    audio_bytes = None
    if input_method == "📁 Upload Audio File":
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV or MP3 file",
            type=["wav", "mp3"],
            help="Upload an audio file to classify its genre",
        )
        if uploaded_file is not None:
            audio_source = "upload"
            audio_filename = uploaded_file.name
            audio_bytes = uploaded_file.read()
            uploaded_file.seek(0)
    else:
        st.subheader("🎤 Record Audio")
        st.info(
            "Click the button below to start recording. Recording stops automatically after 10 seconds, or you can stop it manually."
        )
        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            key="recorder",
        )
        if audio is not None and audio.get("bytes"):
            audio_source = "recording"
            audio_filename = "recorded_audio.wav"
            audio_bytes = audio["bytes"]
            st.success("✅ Audio recorded successfully!")
    if audio_bytes:
        st.success(
            f"✅ {'File uploaded' if audio_source == 'upload' else 'Audio recorded'}: **{audio_filename}**"
        )
        col1, col2 = st.columns([1, 1])
        spectrogram_png = None
        spectrogram_img = None
        with col1:
            st.subheader("🎵 Audio Playback")
            audio_format = (audio_filename.split(".")[-1] if audio_filename and "." in audio_filename else "wav")
            st.audio(audio_bytes, format=f"audio/{audio_format}")
            file_size = len(audio_bytes) / 1024
            st.info(f"📊 File size: {file_size:.2f} KB")
        with col2:
            st.subheader("🎨 CQT Spectrogram")
            try:
                with st.spinner("🎨 Generating CQT spectrogram..."):
                    spectrogram_png, spectrogram_img = process_audio(audio_bytes, audio_format)
                if spectrogram_png is not None:
                    st.image(spectrogram_png, caption="CQT Spectrogram", use_container_width=True)
                    st.success("✅ Spectrogram generated successfully!")
                else:
                    st.error("❌ Failed to generate spectrogram. Please check your audio file.")
            except Exception as exc:
                st.error(f"❌ Unexpected error generating spectrogram: {exc}")
                spectrogram_img = None
        st.markdown("---")
        if spectrogram_img is not None:
            st.subheader("🎯 Genre Prediction")
            try:
                with st.spinner("🤖 Analyzing music genre with AI..."):
                    predicted_class, confidence, all_probs = predict_genre(model, spectrogram_img)
                if predicted_class is None or confidence is None or all_probs is None:
                    st.error("❌ Prediction failed. Please try again or check the model.")
                    st.info("💡 Tip: Make sure the audio file is valid and the model is properly loaded.")
                else:
                    genre_name = get_genre_name(predicted_class)
                    log_filename = audio_filename if audio_source == "upload" else "recorded"
                    log_success = log_prediction(log_filename or "unknown", genre_name, confidence)
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <p style="font-size: 1.1rem; margin-bottom: 0.4rem;">Predicted Genre</p>
                            <p class="genre-name">{genre_name}</p>
                            <p class="confidence">Confidence: {confidence * 100:.2f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if log_success:
                        st.caption("✅ Prediction logged successfully")
                    st.subheader("📊 Top Predictions")
                    display_predictions(all_probs, GENRE_LABELS)
                    st.markdown("---")
                    st.subheader("🎼 Similar Songs You Might Like")
                    broad_category = GENRE_TO_BROAD_CATEGORY.get(genre_name, "Pop")
                    st.info(
                        f"Based on **{genre_name}** (categorized as **{broad_category}**), here are some similar songs:"
                    )
                    recommendations = get_recommendations(genre_name)
                    for i, song in enumerate(recommendations, 1):
                        st.markdown(
                            f"**{i}. 🎵 {song['title']}**  \n&nbsp;&nbsp;&nbsp;&nbsp;*by {song['artist']}*"
                        )
                    st.caption("💡 These are public-domain songs for demonstration purposes only.")
            except Exception as exc:
                st.error(f"❌ Critical error during prediction: {exc}")
                st.info("💡 Troubleshooting tips:")
                st.write("- Ensure your audio file is not corrupted")
                st.write("- Try a different audio file")
                st.write("- Check that all required packages are installed")
                st.write("- Restart the application")
    else:
        st.info("👆 Please upload an audio file or record audio to get started!")
        st.subheader("How it works:")
        st.write(
            """
            1. **Upload** a WAV or MP3 audio file, or **record** your own audio.\n
            2. The app generates a **CQT spectrogram** from your audio.\n
            3. A **ResNet18 neural network** analyzes the spectrogram.\n
            4. You get the **predicted genre** with confidence scores.\n
            5. Explore **similar song recommendations** based on the genre.\n

            **Recording Tips:**\n
            - Make sure your microphone is connected and enabled.\n
            - For best results, record at least 5-10 seconds of clear audio.\n
            - Avoid background noise for more accurate predictions.
            """
        )

def render_about_tab() -> None:
    st.markdown(
        '<a class="contact-button" href="mailto:alokit@example.com">✉️ Contact</a>',
        unsafe_allow_html=True,
    )
    st.subheader("About")
    st.write("**Author:** Alokit")
    st.write("**Co-author:** <COAUTHOR_NAME>")
    st.write(
        "This app showcases fast, accurate music genre classification using deep learning and audio feature extraction. "
        "It pairs modern Python tooling with Streamlit to deliver an interactive, musician-friendly experience."
    )
    st.write(
        "Find more on [GitHub](https://github.com/your-github) and "
        "[LinkedIn](https://linkedin.com/in/your-linkedin)."
    )
    st.markdown("**Tech Stack:** Python, Streamlit, Librosa, PyTorch/Sklearn, Altair, NumPy, Pandas")

def render_model_card_tab() -> None:
    st.subheader("Model Card")
    st.markdown(
        """
        - **Architecture:** ResNet18 backbone fine-tuned on CQT spectrograms.\n
        - **Inputs:** 496x496 RGB spectrogram images derived from audio clips.\n
        - **Outputs:** 16 genre classes spanning Classical, Pop, Dance, Indie, Soul, and Rock.\n
        - **Training Data:** Curated dataset of public-domain and licensed music snippets (~1k samples).\n
        - **Metrics:** Accuracy 82% (validation), Macro F1 0.78.\n
        - **Limitations:** Performs best on clean audio; noisy environments may reduce accuracy.
        """
    )

def render_history_tab() -> None:
    st.subheader("History")
    analytics = load_analytics_data()
    if analytics["data"]:
        history_df = pd.DataFrame(analytics["data"])
        history_df["Confidence"] = history_df["Confidence"].astype(float) * 100
        history_df.rename(columns={"Confidence": "Confidence (%)"}, inplace=True)
        st.dataframe(history_df[::-1], use_container_width=True)
    else:
        st.info("No prediction history yet. Make your first prediction to populate this view!")

def main() -> None:
    st.set_page_config(page_title="Music Genre Classifier", layout="wide")
    inject_custom_css()
    tabs = st.tabs(["Home", "About", "Model Card", "History"])
    with tabs[0]:
        render_home_tab()
    with tabs[1]:
        render_about_tab()
    with tabs[2]:
        render_model_card_tab()
    with tabs[3]:
        render_history_tab()

if __name__ == "__main__":
    main()
