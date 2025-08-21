🖐️ South African Sign Language (SASL) Hand Detection

This project extends a hand detection system (using MediaPipe
) to recognize South African Sign Language (SASL) letters.
It collects landmark data, trains a machine learning model, and predicts SASL hand signs in real time.

🚀 Features

📷 Real-time hand tracking with MediaPipe.

📝 Dataset collection for each SASL letter (currently A–F, left & right hands).

🤖 RandomForestClassifier for sign recognition.

🔄 Easy to extend with new letters or gestures.


SASL-Hand-Detection/
│── data.csv                # Collected hand landmark dataset
│── collect_data.py         # Script to record MediaPipe landmarks
│── train_model.py          # Trains ML model on collected data
│── predict.py              # Real-time SASL prediction
│── requirements.txt        # Python dependencies
│── README.md   

pip install -r requirements.txt
