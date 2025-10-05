import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io


def audio_to_cqt_spectrogram(audio_path, target_size=(496, 496), sr=22050, n_bins=84, bins_per_octave=12):
    """
    Convert an audio file (WAV or MP3) to a CQT spectrogram as a PIL Image.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file (WAV or MP3)
    target_size : tuple
        Target size for the output image (width, height)
    sr : int
        Sample rate for audio loading
    n_bins : int
        Number of frequency bins for CQT
    bins_per_octave : int
        Number of bins per octave for CQT
    
    Returns:
    --------
    PIL.Image
        CQT spectrogram as a PIL Image of size 496x496
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute CQT
    cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    # Convert to decibels (dB)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Normalize to [0, 255] range
    cqt_normalized = ((cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min()) * 255).astype(np.uint8)
    
    # Create PIL Image from the normalized CQT
    # Transpose to get correct orientation (frequency on y-axis, time on x-axis)
    img = Image.fromarray(cqt_normalized)
    
    # Resize to target size (496x496)
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return img_resized


def audio_to_cqt_spectrogram_matplotlib(audio_path, target_size=(496, 496), sr=22050, n_bins=84, bins_per_octave=12):
    """
    Alternative method using matplotlib for more visually appealing spectrograms.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file (WAV or MP3)
    target_size : tuple
        Target size for the output image (width, height)
    sr : int
        Sample rate for audio loading
    n_bins : int
        Number of frequency bins for CQT
    bins_per_octave : int
        Number of bins per octave for CQT
    
    Returns:
    --------
    PIL.Image
        CQT spectrogram as a PIL Image of size 496x496
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute CQT
    cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    # Convert to decibels (dB)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Create figure without axes for clean spectrogram
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Display CQT spectrogram
    librosa.display.specshow(cqt_db, sr=sr, x_axis=None, y_axis=None, ax=ax, cmap='viridis')
    
    # Convert matplotlib figure to PIL Image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    # Resize to target size (496x496)
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    
    return img_resized


# Example usage
if __name__ == "__main__":
    # Example: Convert an audio file to CQT spectrogram
    audio_file = "example_audio.wav"  # Replace with your audio file path
    
    try:
        # Method 1: Direct conversion (faster)
        print("Converting audio to CQT spectrogram (direct method)...")
        spectrogram_img = audio_to_cqt_spectrogram(audio_file)
        spectrogram_img.save("cqt_spectrogram_direct.png")
        print(f"Saved CQT spectrogram as 'cqt_spectrogram_direct.png'")
        print(f"Image size: {spectrogram_img.size}")
        
        # Method 2: Using matplotlib (better visualization)
        print("\nConverting audio to CQT spectrogram (matplotlib method)...")
        spectrogram_img_mpl = audio_to_cqt_spectrogram_matplotlib(audio_file)
        spectrogram_img_mpl.save("cqt_spectrogram_matplotlib.png")
        print(f"Saved CQT spectrogram as 'cqt_spectrogram_matplotlib.png'")
        print(f"Image size: {spectrogram_img_mpl.size}")
        
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_file}' not found.")
        print("Please provide a valid WAV or MP3 file path.")
    except Exception as e:
        print(f"Error processing audio: {e}")
