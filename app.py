import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import csv
from datetime import datetime
from collections import Counter
import pandas as pd
from streamlit_mic_recorder import mic_recorder

# Import custom modules
from audio_to_spectrogram import audio_to_cqt_spectrogram
from load_model import create_genre_classifier, GenreClassifier
from genre_labels import get_genre_name, GENRE_LABELS
from fake_recommendations import get_recommendations, format_recommendations


# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .genre-name {
        font-size: 2rem;
        color: #1DB954;
        font-weight: bold;
    }
    .confidence {
        font-size: 1.2rem;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the pre-trained model. Cached to avoid reloading."""
    try:
        with st.spinner("Loading AI model..."):
            # Try to load a trained checkpoint if it exists
            if os.path.exists('genre_model.pth'):
                st.info("📦 Loading trained model checkpoint...")
                try:
                    model = create_genre_classifier(num_classes=16, pretrained=False, device='cpu')
                    checkpoint = torch.load('genre_model.pth', map_location='cpu')
                    model.load_state_dict(checkpoint)
                    model.eval()
                    st.success("✅ Trained model loaded successfully!")
                    return model
                except Exception as checkpoint_error:
                    st.warning(f"⚠️ Could not load checkpoint: {checkpoint_error}")
                    st.info("Falling back to pre-trained ImageNet model...")
            
            # Use pre-trained ImageNet model with modified final layer
            st.info("🧠 Loading pre-trained ResNet18 model...")
            model = create_genre_classifier(num_classes=16, pretrained=True, device='cpu')
            model.eval()
            st.success("✅ Model loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"❌ Critical error loading model: {e}")
        st.error("Please ensure PyTorch and torchvision are installed correctly.")
        st.code("pip install torch torchvision", language="bash")
        return None


def preprocess_spectrogram(image):
    """
    Preprocess spectrogram image for model input.
    
    Parameters:
    -----------
    image : PIL.Image
        Input spectrogram image (496x496)
    
    Returns:
    --------
    torch.Tensor
        Preprocessed tensor ready for model input
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert to tensor and rearrange dimensions (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def predict_genre(model, spectrogram_image):
    """
    Predict music genre from spectrogram image.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained genre classification model
    spectrogram_image : PIL.Image
        CQT spectrogram image
    
    Returns:
    --------
    tuple
        (predicted_class_id, confidence, all_probabilities) or (None, None, None) on error
    """
    try:
        if model is None:
            st.error("❌ Model is not loaded. Cannot make predictions.")
            return None, None, None
        
        if spectrogram_image is None:
            st.error("❌ No spectrogram image provided.")
            return None, None, None
        
        # Preprocess the image
        try:
            input_tensor = preprocess_spectrogram(spectrogram_image)
        except Exception as preprocess_error:
            st.error(f"❌ Error preprocessing spectrogram: {preprocess_error}")
            return None, None, None
        
        # Make prediction
        with torch.no_grad():
            try:
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                return predicted.item(), confidence.item(), probabilities[0].numpy()
                
            except RuntimeError as runtime_error:
                st.error(f"❌ Model inference error: {runtime_error}")
                st.error("This may be due to incompatible input size or model architecture.")
                return None, None, None
    
    except Exception as e:
        st.error(f"❌ Unexpected error during prediction: {e}")
        st.exception(e)
        return None, None, None


def log_prediction(filename, predicted_genre, confidence, log_file='predictions.log'):
    """
    Log prediction to CSV file with timestamp.
    
    Parameters:
    -----------
    filename : str
        Name of the audio file or 'recorded'
    predicted_genre : str
        Predicted genre name
    confidence : float
        Prediction confidence (0-1)
    log_file : str
        Path to the log file
    """
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(log_file)
        
        # Append to log file
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['Timestamp', 'Filename', 'Predicted_Genre', 'Confidence'])
            
            # Write prediction data
            writer.writerow([timestamp, filename, predicted_genre, f'{confidence:.4f}'])
        
        return True
    except Exception as e:
        st.warning(f"Could not log prediction: {e}")
        return False


def load_analytics_data(log_file='predictions.log'):
    """
    Load and analyze prediction log data.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    
    Returns:
    --------
    dict
        Dictionary containing analytics data
    """
    analytics = {
        'total_predictions': 0,
        'genre_counts': {},
        'recordings': 0,
        'uploads': 0,
        'data': []
    }
    
    if not os.path.exists(log_file):
        return analytics
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                analytics['data'].append(row)
                analytics['total_predictions'] += 1
                
                # Count genres
                genre = row['Predicted_Genre']
                analytics['genre_counts'][genre] = analytics['genre_counts'].get(genre, 0) + 1
                
                # Count recordings vs uploads
                if row['Filename'] == 'recorded':
                    analytics['recordings'] += 1
                else:
                    analytics['uploads'] += 1
        
        return analytics
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        return analytics


def display_analytics():
    """Display analytics dashboard in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Analytics Dashboard")
    
    # Load analytics data
    analytics = load_analytics_data()
    
    if analytics['total_predictions'] == 0:
        st.sidebar.info("No predictions logged yet. Make some predictions to see analytics!")
        return
    
    # Total predictions
    st.sidebar.metric("Total Predictions", analytics['total_predictions'])
    
    # Recordings vs Uploads
    st.sidebar.subheader("📁 Source Breakdown")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("🎤 Recordings", analytics['recordings'])
    with col2:
        st.metric("📂 Uploads", analytics['uploads'])
    
    # Most frequent genres
    st.sidebar.subheader("🎵 Genre Distribution")
    
    if analytics['genre_counts']:
        # Create DataFrame for bar chart
        genre_df = pd.DataFrame(
            list(analytics['genre_counts'].items()),
            columns=['Genre', 'Count']
        ).sort_values('Count', ascending=False)
        
        # Display top 10 genres
        top_genres = genre_df.head(10)
        
        # Show bar chart
        st.sidebar.bar_chart(top_genres.set_index('Genre'))
        
        # Show detailed table
        with st.sidebar.expander("View Detailed Genre Counts"):
            for idx, row in genre_df.iterrows():
                percentage = (row['Count'] / analytics['total_predictions']) * 100
                st.write(f"**{row['Genre']}**: {row['Count']} ({percentage:.1f}%)")
    
    # Recent predictions
    st.sidebar.subheader("🕒 Recent Predictions")
    recent_count = min(5, len(analytics['data']))
    
    if recent_count > 0:
        recent_predictions = analytics['data'][-recent_count:][::-1]  # Last 5, reversed
        
        for pred in recent_predictions:
            with st.sidebar.expander(f"{pred['Predicted_Genre']} ({pred['Timestamp']})"):
                st.write(f"**File:** {pred['Filename']}")
                st.write(f"**Genre:** {pred['Predicted_Genre']}")
                st.write(f"**Confidence:** {float(pred['Confidence']) * 100:.2f}%")
    
    # Download log file
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Download Full Log"):
        try:
            with open('predictions.log', 'r', encoding='utf-8') as f:
                log_content = f.read()
            st.sidebar.download_button(
                label="Download predictions.log",
                data=log_content,
                file_name="predictions.log",
                mime="text/csv"
            )
        except Exception as e:
            st.sidebar.error(f"Could not load log file: {e}")


def process_audio(audio_data, file_extension='wav'):
    """
    Process audio data (either from file upload or recording).
    
    Parameters:
    -----------
    audio_data : bytes or file-like object
        Audio data to process
    file_extension : str
        File extension (wav or mp3)
    
    Returns:
    --------
    PIL.Image or None
        Spectrogram image if successful, None otherwise
    """
    temp_audio_path = f"temp_audio.{file_extension}"
    
    try:
        # Save audio data to temporary file
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            # From mic_recorder
            if not audio_data.get('bytes'):
                st.error("❌ Recording data is empty. Please try recording again.")
                return None
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data['bytes'])
        elif hasattr(audio_data, 'read'):
            # From file uploader
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data.getbuffer())
        else:
            # Direct bytes
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
        
        # Check if file was created and has content
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            st.error("❌ Audio file is empty or could not be saved.")
            return None
        
        # Generate spectrogram
        try:
            spectrogram_img = audio_to_cqt_spectrogram(temp_audio_path, target_size=(496, 496))
            return spectrogram_img
        except FileNotFoundError:
            st.error("❌ Audio file could not be found. Please try again.")
            return None
        except Exception as spec_error:
            st.error(f"❌ Error generating spectrogram: {spec_error}")
            st.error("This may be due to corrupted audio or unsupported format.")
            return None
        
    except PermissionError:
        st.error("❌ Permission denied when saving audio file. Please check file permissions.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error processing audio: {e}")
        st.exception(e)
        return None
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except Exception as cleanup_error:
            st.warning(f"⚠️ Could not remove temporary file: {cleanup_error}")


# Main app
def main():
    # Sidebar - Analytics toggle
    st.sidebar.title("⚙️ Settings")
    show_analytics = st.sidebar.checkbox("Show Analytics (Admin)", value=False)
    
    # Display analytics if enabled
    if show_analytics:
        display_analytics()
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload an audio file to classify its music genre using AI</p>',
        unsafe_allow_html=True
    )
    
    # Description
    with st.expander("ℹ️ About this app", expanded=False):
        st.write("""
        This application uses a deep learning model based on ResNet18 to classify music into 16 different genres:
        
        **Classical**: Symphony, Opera, Solo, Chamber
        
        **Pop**: Pop Vocal Ballad, Adult Contemporary, Teen Pop, Acoustic Pop
        
        **Dance**: Contemporary Dance Pop, Dance Pop
        
        **Indie**: Classic Indie Pop, Chamber Cabaret and Art Pop
        
        **Soul**: Soul or R&B
        
        **Rock**: Adult Alternative Rock, Uplifting Anthemic Rock, Soft Rock
        
        The model analyzes the audio's Constant-Q Transform (CQT) spectrogram to make predictions.
        """)
    
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the error messages above.")
        return
    
    # Input method selection
    st.subheader("🎵 Choose Input Method")
    input_method = st.radio(
        "How would you like to provide audio?",
        ["📁 Upload Audio File", "🎤 Record Audio"],
        horizontal=True
    )
    
    st.markdown("---")
    
    audio_source = None
    audio_data = None
    audio_filename = None
    
    if input_method == "📁 Upload Audio File":
        # File uploader
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV or MP3 file",
            type=["wav", "mp3"],
            help="Upload an audio file to classify its genre"
        )
        
        if uploaded_file is not None:
            audio_source = "upload"
            audio_data = uploaded_file
            audio_filename = uploaded_file.name
    
    else:
        # Audio recorder
        st.subheader("🎤 Record Audio")
        st.info("Click the button below to start recording. The recording will stop automatically after 10 seconds, or you can stop it manually.")
        
        audio = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            key='recorder'
        )
        
        if audio is not None:
            audio_source = "recording"
            audio_data = audio
            audio_filename = "recorded_audio.wav"
            st.success("✅ Audio recorded successfully!")
    
    if audio_data is not None:
        # Display file information
        if audio_source == "upload":
            st.success(f"✅ File uploaded: **{audio_filename}**")
        else:
            st.success(f"✅ Audio recorded: **{audio_filename}**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎵 Audio Playback")
            
            # Play audio
            if audio_source == "upload":
                st.audio(audio_data, format=f'audio/{audio_filename.split(".")[-1]}')
                file_size = audio_data.size / 1024  # Convert to KB
                st.info(f"📊 File size: {file_size:.2f} KB")
            else:
                # For recorded audio
                st.audio(audio_data['bytes'], format='audio/wav')
                file_size = len(audio_data['bytes']) / 1024
                st.info(f"📊 Recording size: {file_size:.2f} KB")
        
        with col2:
            st.subheader("🎨 CQT Spectrogram")
            
            try:
                # Generate spectrogram
                with st.spinner("🎨 Generating CQT spectrogram..."):
                    file_extension = audio_filename.split('.')[-1]
                    spectrogram_img = process_audio(audio_data, file_extension)
                
                if spectrogram_img is not None:
                    # Display spectrogram
                    st.image(spectrogram_img, caption="CQT Spectrogram (496x496)", use_container_width=True)
                    st.success("✅ Spectrogram generated successfully!")
                else:
                    st.error("❌ Failed to generate spectrogram. Please check your audio file.")
                    spectrogram_img = None
                
            except Exception as e:
                st.error(f"❌ Unexpected error generating spectrogram: {e}")
                st.error("Please try with a different audio file or check file format.")
                spectrogram_img = None
        
        st.markdown("---")
        
        # Make prediction
        if spectrogram_img is not None:
            st.subheader("🎯 Genre Prediction")
            
            try:
                with st.spinner("🤖 Analyzing music genre with AI..."):
                    predicted_class, confidence, all_probs = predict_genre(model, spectrogram_img)
                
                # Check if prediction was successful
                if predicted_class is None or confidence is None or all_probs is None:
                    st.error("❌ Prediction failed. Please try again or check the model.")
                    st.info("💡 Tip: Make sure the audio file is valid and the model is properly loaded.")
                else:
                    genre_name = get_genre_name(predicted_class)
                    
                    # Log the prediction
                    try:
                        log_filename = audio_filename if audio_source == "upload" else "recorded"
                        log_success = log_prediction(log_filename, genre_name, confidence)
                    except Exception as log_error:
                        st.warning(f"⚠️ Could not log prediction: {log_error}")
                        log_success = False
                    
                    # Display prediction
                    st.success("✅ Genre prediction completed!")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="font-size: 1.2rem; margin-bottom: 10px;">Predicted Genre:</p>
                        <p class="genre-name">{genre_name}</p>
                        <p class="confidence">Confidence: {confidence * 100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show logging status
                    if log_success:
                        st.caption("✅ Prediction logged successfully")
                    
                    # Show top 5 predictions
                    st.subheader("📊 Top 5 Genre Predictions")
                    
                    try:
                        top5_indices = np.argsort(all_probs)[-5:][::-1]
                        
                        for i, idx in enumerate(top5_indices):
                            genre = get_genre_name(idx)
                            prob = all_probs[idx] * 100
                            
                            # Create progress bar
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.progress(float(all_probs[idx]))
                            with col_b:
                                st.write(f"**{genre}**")
                            st.write(f"{prob:.2f}%")
                            st.markdown("<br>", unsafe_allow_html=True)
                    except Exception as chart_error:
                        st.warning(f"⚠️ Could not display probability chart: {chart_error}")
                    
                    # Show song recommendations
                    st.markdown("---")
                    st.subheader("🎼 Similar Songs You Might Like")
                    
                    try:
                        # Import the mapping to show broad category
                        from fake_recommendations import GENRE_TO_BROAD_CATEGORY
                        
                        # Get broad genre category
                        broad_category = GENRE_TO_BROAD_CATEGORY.get(genre_name, 'Pop')
                        st.info(f"Based on **{genre_name}** (categorized as **{broad_category}**), here are some similar songs:")
                        
                        # Get and display recommendations with better formatting
                        recommendations = get_recommendations(genre_name)
                        
                        for i, song in enumerate(recommendations, 1):
                            st.markdown(f"""
                            **{i}. 🎵 {song['title']}**  
                            &nbsp;&nbsp;&nbsp;&nbsp;*by {song['artist']}*
                            """)
                        
                        st.caption("💡 These are public-domain songs for demonstration purposes only.")
                    except Exception as rec_error:
                        st.warning(f"⚠️ Could not load recommendations: {rec_error}")
                    
            except Exception as e:
                st.error(f"❌ Critical error during prediction: {e}")
                st.error("Please try again or contact support if the issue persists.")
                st.exception(e)
                st.info("💡 Troubleshooting tips:")
                st.write("- Ensure your audio file is not corrupted")
                st.write("- Try a different audio file")
                st.write("- Check that all required packages are installed")
                st.write("- Restart the application")
    
    else:
        # Placeholder when no audio is provided
        st.info("👆 Please upload an audio file or record audio to get started!")
        
        # Show example spectrograms or instructions
        st.subheader("How it works:")
        st.write("""
        1. **Upload** a WAV or MP3 audio file, or **record** your own audio
        2. The app generates a **CQT spectrogram** from your audio
        3. A **ResNet18 neural network** analyzes the spectrogram
        4. You get the **predicted genre** with confidence scores
        5. Explore **similar song recommendations** based on the genre
        
        **Recording Tips:**
        - Make sure your microphone is connected and enabled
        - For best results, record at least 5-10 seconds of clear audio
        - Avoid background noise for more accurate predictions
        """)


if __name__ == "__main__":
    main()
