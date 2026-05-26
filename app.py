import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from deepface import DeepFace


st.set_page_config(
    page_title="Mood-Based Song Recommender",
    layout="wide"
)


def load_song_dataset():
    songs = pd.DataFrame({
        "title": [
            "Weightless",
            "Blinding Lights",
            "Believer",
            "Fix You",
            "Levitating",
            "Someone Like You",
            "Perfect",
            "Heat Waves",
            "Let Her Go",
            "Closer",
            "Stay",
            "Shivers",
            "Numb",
            "Photograph",
            "Night Changes"
        ],
        "artist": [
            "Marconi Union",
            "The Weeknd",
            "Imagine Dragons",
            "Coldplay",
            "Dua Lipa",
            "Adele",
            "Ed Sheeran",
            "Glass Animals",
            "Passenger",
            "The Chainsmokers",
            "The Kid LAROI & Justin Bieber",
            "Ed Sheeran",
            "Linkin Park",
            "Ed Sheeran",
            "One Direction"
        ],
        "genre": [
            "Ambient",
            "Pop",
            "Rock",
            "Soft Rock",
            "Pop",
            "Ballad",
            "Romantic",
            "Indie",
            "Acoustic",
            "EDM",
            "Pop",
            "Pop",
            "Rock",
            "Romantic",
            "Pop"
        ],
        "relaxing":   [1.0, 0.2, 0.1, 0.7, 0.3, 0.6, 0.8, 0.5, 0.7, 0.4, 0.2, 0.3, 0.1, 0.8, 0.6],
        "uplifting":  [0.2, 0.9, 0.7, 0.4, 0.9, 0.2, 0.6, 0.7, 0.3, 0.8, 0.9, 0.9, 0.3, 0.4, 0.5],
        "intense":    [0.0, 0.7, 0.9, 0.2, 0.5, 0.1, 0.2, 0.5, 0.2, 0.6, 0.8, 0.6, 0.9, 0.2, 0.3],
        "melancholic":[0.2, 0.2, 0.3, 0.8, 0.1, 0.9, 0.5, 0.4, 0.9, 0.2, 0.2, 0.2, 0.8, 0.7, 0.6],
        "focus":      [0.9, 0.3, 0.2, 0.6, 0.4, 0.5, 0.5, 0.4, 0.6, 0.3, 0.3, 0.4, 0.2, 0.7, 0.5],
        "youtube_url": [
            "https://www.youtube.com/watch?v=UfcAVejslrU",
            "https://www.youtube.com/watch?v=4NRXx6U8ABQ",
            "https://www.youtube.com/watch?v=7wtfhZwyrcc",
            "https://www.youtube.com/watch?v=k4V3Mo61fJM",
            "https://www.youtube.com/watch?v=TUVcZfQe-Kw",
            "https://www.youtube.com/watch?v=hLQl3WQQoQ0",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=mRD0-GxqHVo",
            "https://www.youtube.com/watch?v=RBumgq5yVrA",
            "https://www.youtube.com/watch?v=PT2_F-1esPk",
            "https://www.youtube.com/watch?v=kTJczUoc26U",
            "https://www.youtube.com/watch?v=Il0S8BoucSA",
            "https://www.youtube.com/watch?v=kXYiU_JCYtU",
            "https://www.youtube.com/watch?v=nSDgHBxUbVQ",
            "https://www.youtube.com/watch?v=syFZfO_wfMQ"
        ]
    })
    return songs


songs = load_song_dataset()

def build_fuzzy_system():
    valence = ctrl.Antecedent(np.arange(0, 11, 1), "valence")
    energy = ctrl.Antecedent(np.arange(0, 11, 1), "energy")
    stress = ctrl.Antecedent(np.arange(0, 11, 1), "stress")

    relaxing = ctrl.Consequent(np.arange(0, 11, 1), "relaxing")
    uplifting = ctrl.Consequent(np.arange(0, 11, 1), "uplifting")
    intense = ctrl.Consequent(np.arange(0, 11, 1), "intense")
    melancholic = ctrl.Consequent(np.arange(0, 11, 1), "melancholic")
    focus = ctrl.Consequent(np.arange(0, 11, 1), "focus")

    for var in [valence, energy, stress]:
        var["low"] = fuzz.trimf(var.universe, [0, 0, 5])
        var["medium"] = fuzz.trimf(var.universe, [2, 5, 8])
        var["high"] = fuzz.trimf(var.universe, [5, 10, 10])

    for var in [relaxing, uplifting, intense, melancholic, focus]:
        var["low"] = fuzz.trimf(var.universe, [0, 0, 5])
        var["medium"] = fuzz.trimf(var.universe, [2, 5, 8])
        var["high"] = fuzz.trimf(var.universe, [5, 10, 10])

    rules = [
        ctrl.Rule(valence["low"] & stress["high"], relaxing["high"]),
        ctrl.Rule(valence["high"] & energy["high"], uplifting["high"]),
        ctrl.Rule(stress["high"] & energy["high"], intense["high"]),
        ctrl.Rule(valence["low"] & energy["low"], melancholic["high"]),
        ctrl.Rule(valence["medium"] & energy["low"], focus["medium"]),
        ctrl.Rule(stress["low"] & energy["medium"], uplifting["medium"]),
        ctrl.Rule(energy["low"] & stress["high"], relaxing["high"]),
        ctrl.Rule(valence["medium"] & energy["medium"], focus["medium"]),
    ]

    return ctrl.ControlSystem(rules)


fuzzy_ctrl = build_fuzzy_system()


def run_fuzzy_inference(valence_input, energy_input, stress_input):
    sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    sim.input["valence"] = float(valence_input)
    sim.input["energy"] = float(energy_input)
    sim.input["stress"] = float(stress_input)
    sim.compute()

    return {
        "relaxing": sim.output.get("relaxing", 0.0),
        "uplifting": sim.output.get("uplifting", 0.0),
        "intense": sim.output.get("intense", 0.0),
        "melancholic": sim.output.get("melancholic", 0.0),
        "focus": sim.output.get("focus", 0.0),
    }

def sugeno_score(valence, energy, stress):
    rules = []

    if valence <= 5 and stress >= 5:
        rules.append((0.8, 8.5))
    if valence >= 5 and energy >= 5:
        rules.append((0.7, 9.0))
    if stress >= 5 and energy >= 5:
        rules.append((0.6, 7.5))

    if not rules:
        return 5.0

    weighted_sum = sum(w * z for w, z in rules)
    total_weight = sum(w for w, z in rules)
    return weighted_sum / total_weight


def map_emotion_to_mood(emotions):
    happy = emotions.get("happy", 0.0)
    sad = emotions.get("sad", 0.0)
    angry = emotions.get("angry", 0.0)
    neutral = emotions.get("neutral", 0.0)
    surprise = emotions.get("surprise", 0.0)
    fear = emotions.get("fear", 0.0)
    disgust = emotions.get("disgust", 0.0)

    total = happy + sad + angry + neutral + surprise + fear + disgust
    if total == 0:
        total = 1.0

    happy /= total
    sad /= total
    angry /= total
    neutral /= total
    surprise /= total
    fear /= total
    disgust /= total

    valence = (
        happy * 9.0 +
        neutral * 5.0 +
        surprise * 6.0 +
        sad * 2.0 +
        angry * 1.5 +
        fear * 1.5 +
        disgust * 1.0
    )

    energy = (
        happy * 7.5 +
        neutral * 4.0 +
        surprise * 8.5 +
        sad * 3.0 +
        angry * 8.5 +
        fear * 7.5 +
        disgust * 6.0
    )

    stress = (
        happy * 2.0 +
        neutral * 4.0 +
        surprise * 6.5 +
        sad * 7.0 +
        angry * 9.0 +
        fear * 8.5 +
        disgust * 7.5
    )

    return round(valence, 2), round(energy, 2), round(stress, 2)


def analyze_face_emotion(pil_image):
    try:
        img_array = np.array(pil_image)

        result = DeepFace.analyze(
            img_path=img_array,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})
        dominant_emotion = result.get("dominant_emotion", "unknown")

        return dominant_emotion, emotions, None
    except Exception as e:
        return None, None, str(e)


def calculate_fuzzy_score(row, target):
    score = 0.0
    score += abs(row["relaxing"] * 10 - target["relaxing"]) * 0.30
    score += abs(row["uplifting"] * 10 - target["uplifting"]) * 0.25
    score += abs(row["intense"] * 10 - target["intense"]) * 0.20
    score += abs(row["melancholic"] * 10 - target["melancholic"]) * 0.15
    score += abs(row["focus"] * 10 - target["focus"]) * 0.10
    return score


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=30):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        y = y.reshape(-1, 1)

        for _ in range(epochs):
            hidden_input = np.dot(X, self.W1)
            hidden_output = self.sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.W2)
            final_output = self.sigmoid(final_input)

            error = y - final_output
            d_output = error * self.sigmoid_derivative(final_output)

            hidden_error = d_output.dot(self.W2.T)
            d_hidden = hidden_error * self.sigmoid_derivative(hidden_output)

            self.W2 += hidden_output.T.dot(d_output) * learning_rate
            self.W1 += X.T.dot(d_hidden) * learning_rate

    def predict(self, X):
        hidden_output = self.sigmoid(np.dot(X, self.W1))
        final_output = self.sigmoid(np.dot(hidden_output, self.W2))
        return final_output


def build_nn_scores(song_df):
    X_train = song_df[["relaxing", "uplifting", "intense", "melancholic", "focus"]].values
    y_train = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    perceptron = Perceptron(learning_rate=0.1, epochs=30)
    perceptron.fit(X_train, y_train)
    perceptron_predictions = perceptron.predict(X_train)
    perceptron_acc = accuracy_score(y_train, perceptron_predictions)

    nn_model = SimpleNeuralNetwork(input_size=5, hidden_size=4)
    nn_model.train(X_scaled, y_train, epochs=2000, learning_rate=0.1)
    nn_predictions = nn_model.predict(X_scaled).flatten()
    nn_binary_predictions = (nn_predictions > 0.5).astype(int)
    nn_acc = accuracy_score(y_train, nn_binary_predictions)

    return nn_predictions, perceptron_acc, nn_acc

def show_membership_plot():
    x = np.arange(0, 11, 1)
    low = fuzz.trimf(x, [0, 0, 5])
    medium = fuzz.trimf(x, [2, 5, 8])
    high = fuzz.trimf(x, [5, 10, 10])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, low, label="Low")
    ax.plot(x, medium, label="Medium")
    ax.plot(x, high, label="High")
    ax.set_title("Membership Functions for Mood Inputs")
    ax.set_xlabel("Mood Scale (0-10)")
    ax.set_ylabel("Membership Degree")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def show_output_chart(fuzzy_output):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(fuzzy_output.keys()), list(fuzzy_output.values()))
    ax.set_title("Fuzzy Output Music Profile")
    ax.set_ylabel("Score (0-10)")
    ax.grid(axis="y")
    st.pyplot(fig)


st.title("🎵 Hybrid Mood-Based Song Recommendation")
st.markdown("### Fuzzy Logic + Neural Network + Facial Emotion Analysis")

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio(
        "Choose mood input mode",
        ["Face Detection", "Manual Input"]
    )
    show_theory = st.checkbox("Show membership function plot", value=False)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

if show_theory:
    st.subheader("Membership Functions")
    show_membership_plot()

left_col, right_col = st.columns([1.1, 1])

detected_valence, detected_energy, detected_stress = 5.0, 5.0, 5.0
dominant_emotion = None
emotions = None

with left_col:
    st.subheader("Input Section")

    if input_mode == "Face Detection":
        st.write("Capture an image or upload one for facial emotion analysis.")

        cam_image = st.camera_input("Take a picture")
        uploaded_file = st.file_uploader("Or upload a face image", type=["jpg", "jpeg", "png"])

        selected_image = None

        if cam_image is not None:
            selected_image = Image.open(cam_image).convert("RGB")
        elif uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")

        if selected_image is not None:
            st.image(selected_image, caption="Selected Image", use_container_width=True)

            if st.button("Detect Mood from Face"):
                with st.spinner("Analyzing facial emotion..."):
                    dominant_emotion, emotions, error = analyze_face_emotion(selected_image)

                if error:
                    st.error(f"Emotion detection failed: {error}")
                else:
                    detected_valence, detected_energy, detected_stress = map_emotion_to_mood(emotions)

                    st.session_state["detected_valence"] = detected_valence
                    st.session_state["detected_energy"] = detected_energy
                    st.session_state["detected_stress"] = detected_stress
                    st.session_state["dominant_emotion"] = dominant_emotion
                    st.session_state["emotions"] = emotions

        if "detected_valence" in st.session_state:
            detected_valence = st.session_state["detected_valence"]
            detected_energy = st.session_state["detected_energy"]
            detected_stress = st.session_state["detected_stress"]
            dominant_emotion = st.session_state.get("dominant_emotion")
            emotions = st.session_state.get("emotions")

        st.markdown("#### Detected / Editable Mood Values")
        user_valence = st.slider("Valence", 0.0, 10.0, float(detected_valence), 0.1)
        user_energy = st.slider("Energy", 0.0, 10.0, float(detected_energy), 0.1)
        user_stress = st.slider("Stress", 0.0, 10.0, float(detected_stress), 0.1)

    else:
        st.write("Set mood manually.")
        user_valence = st.slider("Valence", 0.0, 10.0, 5.0, 0.1)
        user_energy = st.slider("Energy", 0.0, 10.0, 5.0, 0.1)
        user_stress = st.slider("Stress", 0.0, 10.0, 5.0, 0.1)

with right_col:
    st.subheader("Detected Emotion")
    if dominant_emotion:
        st.success(f"Dominant emotion: **{dominant_emotion}**")
    else:
        st.info("No face-based emotion detected yet.")

    if emotions:
        emotion_df = pd.DataFrame({
            "Emotion": list(emotions.keys()),
            "Confidence": list(emotions.values())
        }).sort_values("Confidence", ascending=False)

        st.dataframe(emotion_df, use_container_width=True)
        st.bar_chart(emotion_df.set_index("Emotion"))
    else:
        st.write("Emotion probabilities will appear here.")

st.divider()

if st.button("Generate Recommendations", type="primary"):
    fuzzy_output = run_fuzzy_inference(user_valence, user_energy, user_stress)
    sugeno_result = sugeno_score(user_valence, user_energy, user_stress)

    songs_copy = songs.copy()
    songs_copy["fuzzy_score"] = songs_copy.apply(
        lambda row: calculate_fuzzy_score(row, fuzzy_output), axis=1
    )

    nn_predictions, perceptron_acc, nn_acc = build_nn_scores(songs_copy)
    songs_copy["nn_score"] = nn_predictions

    songs_copy["hybrid_score"] = (
        0.7 * (1 / (1 + songs_copy["fuzzy_score"])) +
        0.3 * songs_copy["nn_score"]
    )

    final_recommendations = songs_copy.sort_values("hybrid_score", ascending=False).head(top_n)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Mood Summary")
        st.write(f"**Valence:** {user_valence}")
        st.write(f"**Energy:** {user_energy}")
        st.write(f"**Stress:** {user_stress}")
        st.write(f"**Sugeno Score:** {round(sugeno_result, 2)}")

    with c2:
        st.subheader("Neural Model Performance")
        st.write(f"**Perceptron Accuracy:** {round(perceptron_acc, 3)}")
        st.write(f"**Simple NN Accuracy:** {round(nn_acc, 3)}")

    st.subheader("Fuzzy Output Music Profile")
    show_output_chart(fuzzy_output)

    st.subheader("Top Recommended Songs")
    st.dataframe(
        final_recommendations[
            ["title", "artist", "genre", "fuzzy_score", "nn_score", "hybrid_score"]
        ].reset_index(drop=True),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(final_recommendations["title"], final_recommendations["hybrid_score"])
    ax.set_title("Top Recommended Songs by Hybrid Score")
    ax.set_xlabel("Song")
    ax.set_ylabel("Hybrid Score")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y")
    st.pyplot(fig)

    st.subheader("Recommended Songs with Play Option")
    for _, row in final_recommendations.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['title']}")
            st.write(f"**Artist:** {row['artist']}")
            st.write(f"**Genre:** {row['genre']}")
            st.write(f"**Hybrid Score:** {row['hybrid_score']:.4f}")

            reason_parts = []
            max_output = max(fuzzy_output.values())

            if fuzzy_output["relaxing"] == max_output:
                reason_parts.append("calming and soothing mood")
            if fuzzy_output["uplifting"] == max_output:
                reason_parts.append("uplifting mood")
            if fuzzy_output["intense"] == max_output:
                reason_parts.append("high-energy intense mood")
            if fuzzy_output["melancholic"] == max_output:
                reason_parts.append("melancholic emotional tone")
            if fuzzy_output["focus"] == max_output:
                reason_parts.append("focus-oriented mood")

            if not reason_parts:
                reason_parts.append("overall mood similarity")

            st.write("**Why recommended:** Matches your " + ", ".join(reason_parts) + ".")

            if pd.notna(row["youtube_url"]) and str(row["youtube_url"]).strip():
                st.markdown(f"[▶ Play / Watch on YouTube]({row['youtube_url']})")
                st.video(row["youtube_url"])
            else:
                st.warning("No YouTube link available for this song.")

st.divider()
st.caption("Built with Streamlit, fuzzy logic, neural personalization, and facial emotion analysis.")