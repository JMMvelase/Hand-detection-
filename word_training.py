import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import random


DATA_DIR = "sign_words"
WORDS = ["hello", "thanks", "help", "me", "please"]  # update to 5 words
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 42

def normalize_landmarks(sequence):
    # sequence shape: (SEQUENCE_LENGTH, NUM_LANDMARKS)
    # Assume even indices are x, odd are y
    norm_seq = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        frame = sequence[i]
        xs = frame[::2]
        ys = frame[1::2]
        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        norm_xs = (xs - min_x) / (max_x - min_x + 1e-6)
        norm_ys = (ys - min_y) / (max_y - min_y + 1e-6)
        norm_frame = np.empty_like(frame)
        norm_frame[::2] = norm_xs
        norm_frame[1::2] = norm_ys
        norm_seq[i] = norm_frame
    return norm_seq

def add_velocity(sequence):
    # sequence shape: (SEQUENCE_LENGTH, NUM_LANDMARKS)
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    # Concatenate position and velocity features
    return np.concatenate([sequence, velocity], axis=1)

sequences = []
labels = []

for idx, word in enumerate(WORDS):
    word_dir = os.path.join(DATA_DIR, word)
    for file in os.listdir(word_dir):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(word_dir, file))
            seq = normalize_landmarks(seq)
            seq = add_velocity(seq)
            sequences.append(seq)
            labels.append(idx)  # integer label

X = np.array(sequences)   # shape: (num_samples, SEQUENCE_LENGTH, NUM_LANDMARKS*2)
y = np.array(labels)      # shape: (num_samples,)
print("X shape:", X.shape, "y shape:", y.shape)

y_cat = to_categorical(y, num_classes=len(WORDS))

# Shuffle and split dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y_cat = y_cat[indices]
y = y[indices]

num_samples = len(X)
train_end = int(0.7 * num_samples)
val_end = int(0.85 * num_samples)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y_cat[:train_end], y_cat[train_end:val_end], y_cat[val_end:]

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(SEQUENCE_LENGTH, NUM_LANDMARKS*2)))
model.add(LSTM(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(WORDS), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Pick a random test sequence
idx = random.randint(0, len(X_test)-1)
sample = np.expand_dims(X_test[idx], axis=0)
pred = model.predict(sample)
pred_word = WORDS[np.argmax(pred)]
confidence = np.max(pred)
print(f"Predicted: {pred_word} (confidence: {confidence:.2f})")

# Save model with versioned name
import datetime
version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"sign_model_{version}.h5")