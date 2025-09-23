import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import random


DATA_DIR = "sign_words"
WORDS = ["hello", "thanks"]  # same order matters
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 42

sequences = []
labels = []

for idx, word in enumerate(WORDS):
    word_dir = os.path.join(DATA_DIR, word)
    for file in os.listdir(word_dir):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(word_dir, file))
            sequences.append(seq)
            labels.append(idx)  # integer label

# Convert to numpy arrays
X = np.array(sequences)   # shape: (num_samples, SEQUENCE_LENGTH, NUM_LANDMARKS)
y = np.array(labels)      # shape: (num_samples,)
print(X.shape, y.shape)

y = to_categorical(y, num_classes=len(WORDS))
# Shuffle the dataset
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(SEQUENCE_LENGTH, NUM_LANDMARKS)))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(WORDS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=50, batch_size=8, validation_split=0.2)

# Pick a random sequence
idx = random.randint(0, len(X)-1)
sample = np.expand_dims(X[idx], axis=0)  # shape (1, SEQUENCE_LENGTH, NUM_LANDMARKS)

pred = model.predict(sample)
pred_word = WORDS[np.argmax(pred)]
print("Predicted:", pred_word)



##model.save("sasl_model.h5")