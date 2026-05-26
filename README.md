# Emotion-Aware-Music-Recommender

Emotion-Aware-Music-Recommender is an AI-powered music recommendation system that generates personalized song suggestions based on the user's emotional state. The project combines facial emotion recognition, fuzzy logic, neural networks, and interactive visualizations to create a smart and adaptive music recommendation experience.

The system detects emotions using facial analysis or manual mood input and recommends songs that match the user’s emotional condition in real time.

---

# Features

- Facial emotion detection using DeepFace
- Manual mood input support
- Mood-based song recommendation system
- Fuzzy logic-based mood analysis
- Neural network-enhanced recommendation scoring
- Interactive Streamlit web interface
- Real-time visualizations and charts
- YouTube playback link integration
- Hybrid recommendation system combining AI and fuzzy reasoning

---

# Emotion Detection Features

The system detects emotions such as:

- Happy
- Sad
- Angry
- Fear
- Neutral
- Surprise
- Disgust

These emotions are converted into mood parameters:

- Valence
- Energy
- Stress

---

# Recommendation Categories

The fuzzy inference system generates recommendation profiles such as:

- Relaxing
- Uplifting
- Intense
- Melancholic
- Focus

---

# Tech Stack

- Python
- Streamlit
- DeepFace
- TensorFlow / Keras
- scikit-fuzzy
- scikit-learn
- Pandas
- NumPy
- Matplotlib

---

# AI & Logic Components

### Facial Emotion Recognition
Uses DeepFace to analyze facial expressions and extract emotional probabilities.

### Fuzzy Logic System
Processes mood parameters using fuzzy inference rules.

### Neural Network Module
Uses perceptron and neural network models to improve recommendation quality.

### Hybrid Recommendation Engine
Combines:
- Fuzzy similarity scores
- Neural network predictions
- Mood analysis

---

# How It Works

1. User uploads/captures a face image or enters mood manually.
2. DeepFace detects emotions.
3. Emotions are mapped into mood parameters.
4. Fuzzy logic processes emotional inputs.
5. Neural networks calculate recommendation scores.
6. Songs are ranked and recommended.
7. Results are displayed with visualizations and YouTube playback links.

---

# Installation

```bash
git clone https://github.com/your-username/Emotion-Aware-Music-Recommender.git
cd Emotion-Aware-Music-Recommender
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

---

# Project Structure

```bash
Emotion-Aware-Music-Recommender/
├── app.py
├── requirements.txt
├── dataset/
├── visualizations/
├── models/
├── README.md
└── assets/
```

---

# Visual Outputs

The application provides:

- Emotion confidence charts
- Mood analysis graphs
- Membership function plots
- Recommendation score visualizations
- Song recommendation rankings

---

# Educational Purpose

This project demonstrates practical implementation of:

- Artificial Intelligence
- Emotion Recognition
- Fuzzy Logic Systems
- Neural Networks
- Recommendation Systems
- Interactive Data Visualization
- Streamlit Application Development

---

# Future Improvements

- Spotify API integration
- Apple Music integration
- Real-time webcam emotion tracking
- Voice emotion analysis
- Reinforcement learning-based recommendations
- Personalized user profiles
- Cloud deployment support
- Multi-language recommendation system

---

# Disclaimer

This project is developed for educational and research purposes only.

---

# License

MIT License
