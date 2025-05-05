# 🎵 Music Genre Classification System

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated web application that leverages machine learning to automatically classify music tracks into different genres. The system analyzes audio files using advanced signal processing techniques and deep learning to provide accurate genre predictions through an intuitive web interface.

## ✨ Features

- **Audio Analysis**: Process and extract features from music files using Librosa
- **ML-Based Classification**: Predict music genres with a trained deep learning model
- **User-Friendly Interface**: Simple web interface for uploading and analyzing tracks
- **Multi-Platform Deployment**: Ready for Vercel, Heroku, or custom server deployment

## 🛠️ Technology Stack

### Backend
- **Python** - Core programming language
- **Flask** - Web application framework
- **Librosa** - Audio feature extraction and analysis
- **TensorFlow/Keras** - Deep learning model architecture
- **Scikit-learn** - Data preprocessing and scaling

### Frontend
- **HTML/CSS** - Responsive user interface
- **Custom Styling** - Enhanced user experience via static/style.css

### Deployment
- **Vercel** - Primary deployment platform (configured via vercel.json)
- **Heroku** - Alternative deployment option (Procfile included)
- **Gunicorn** - WSGI HTTP server for production

## 📋 Prerequisites

- Python 3.6 or higher
- pip package manager
- Basic understanding of machine learning concepts

## 🚀 Getting Started

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ALOK-Yeager/music-genre-classifier-BPIT.git
   cd music-genre-classifier-BPIT
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Create a `.env` file in the project root
   - Add necessary configuration variables (refer to sample.env if provided)

### Running the Application

#### Development Mode
```bash
python app.py
# or
python index.py
```

#### Production Mode
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

The application will be accessible at http://localhost:5000

## 📊 Project Structure

```
music-genre-classifier-BPIT/
├── app.py              # Main Flask application
├── index.py            # Alternative entry point
├── wsgi.py             # WSGI entry point for production
├── requirements.txt    # Python dependencies
├── model/              # Trained ML model files
├── static/             # CSS, JavaScript, and assets
├── templates/          # HTML templates
├── utils/              # Helper functions and utilities
├── notebooks/          # Jupyter notebooks for development
│   └── musicgenreclassification (1).ipynb
├── Procfile            # Heroku deployment configuration
├── vercel.json         # Vercel deployment configuration
└── build.sh            # Build script for deployment
```

## 🔍 Development Process

The initial model exploration and training were conducted in Jupyter notebooks. The notebooks demonstrate the data preprocessing pipeline, feature extraction methodology, model architecture selection, and evaluation metrics.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For any questions or feedback, please reach out to [alokitmishra9899@gmail.com](alokitmishra9899@gmail.com)

---

*Note: This project was developed as part of coursework at BPIT.*
