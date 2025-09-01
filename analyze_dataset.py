import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('sasl_dataset/sasl_landmarks.csv')

# Count samples per letter
letter_counts = df['label'].value_counts()

# Plot distribution
plt.figure(figsize=(15, 5))
plt.bar(letter_counts.index, letter_counts.values)
plt.title('Number of Samples per Letter')
plt.xlabel('Letter')
plt.ylabel('Number of Samples')
plt.show()

# Print statistics
print("\nSample distribution:")
for letter, count in letter_counts.items():
    print(f"Letter {letter}: {count} samples")

print("\nTotal samples:", len(df))
print("Average samples per letter:", len(df) / 26)  # 26 letters
print("\nLetters with few samples:")
for letter, count in letter_counts.items():
    if count < len(df) / (26 * 2):  # Less than half the average
        print(f"Letter {letter}: {count} samples")
