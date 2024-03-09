import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import scrolledtext
from concurrent.futures import ThreadPoolExecutor

nltk.download('stopwords')

# Metin önişleme fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# Metin analizi işlemi
def analyze_text(text):
    words = preprocess_text(text)
    word_freq = Counter(words)
    # VADER duygu analizi
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    polarity = sentiment_scores['compound']  # Compound score kullanarak genel duygu polaritesi
    return word_freq, polarity

# Paralel metin analizi işlemi
def parallel_text_analysis(texts):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_text, texts))
    return results

# Metin analizi ve görselleştirme işlemi
def analyze_and_visualize_text():
    input_texts = input_text.get("1.0", tk.END).splitlines()
    results = parallel_text_analysis(input_texts)

    for i, (word_freq, polarity) in enumerate(results):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(word_freq.keys(), word_freq.values())
        plt.title(f"Text {i + 1} Word Frequency Analysis")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.bar(["Positive", "Negative"], [polarity, -polarity], color=['green', 'red'])
        plt.title(f"Text {i + 1} Sentiment Analysis")
        plt.xlabel("Sentiment")
        plt.ylabel("Polarity")
        plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show()

# Tkinter GUI oluşturma
root = tk.Tk()
root.title("Text Analiz Platformu")

# Metin girişi için bir alan oluştur
input_text = scrolledtext.ScrolledText(root, width=50, height=10, wrap=tk.WORD)
input_text.pack(padx=10, pady=10)

# Analiz düğmesi
analyze_button = tk.Button(root, text="Analiz Yap", command=analyze_and_visualize_text)
analyze_button.pack(pady=5)

root.mainloop()



