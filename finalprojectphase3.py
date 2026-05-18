import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        combined_features = np.hstack((mfccs_processed, chroma, mel))
        return combined_features
    except Exception as e:
        print(f"Hata: {file_path} işlenemedi. {e}")
        return None

DATA_PATH = "/Users/ibrahimcavdar/Desktop/dersler/Signals and Sistems/Final project/Dataset"

emotion_mapping = {
    'neutral': 'Neutral', 'notr': 'Neutral', 'nötr': 'Neutral',
    'happy': 'Happy', 'mutlu': 'Happy',
    'angry': 'Angry', 'furious': 'Angry', 'ofkeli': 'Angry', 'öfkeli': 'Angry',
    'sad': 'Sad', 'uzgun': 'Sad', 'üzgün': 'Sad', 'mutsuz': 'Sad',
    'surprised': 'Surprised', 'shocked': 'Surprised', 'saskin': 'Surprised', 'şaşkın': 'Surprised'
}

X = []
y = []

print("Ses dosyaları taranıyor ve öznitelikler çıkarılıyor...\n")

if os.path.exists(DATA_PATH):
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                filename_lower = file.lower()
                
                target_emotion = None
                for key, emotion_label in emotion_mapping.items():
                    if key in filename_lower:
                        target_emotion = emotion_label
                        break 
                
                if target_emotion is not None:
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(target_emotion)
else:
    print(f"HATA: '{DATA_PATH}' bulunamadı.")

# 3. MODELİ EĞİT VE GÖRSEL ARAYÜZ (UI) İLE SUN
if len(X) > 0:
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Hata Matrisini Hesapla
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"İşlem Tamam! Toplam {len(X)} dosya işlendi.")
    print(f"Doğruluk Skoru: %{accuracy * 100:.2f}. Şimdi ekrana grafik penceresi açılıyor...")

    # --- GÖRSEL ARAYÜZ (UI) OLUŞTURMA KISMI ---
    # Grafik penceresinin boyutunu ayarla
    plt.figure(figsize=(10, 7))
    
    # Isı haritasını (Heatmap) çiz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, 
                annot_kws={"size": 14}) # Rakamların boyutu

    # Grafiğin başlığına Accuracy (Doğruluk) skorunu ekle
    plt.title(f'Sesli Duygu Sınıflandırma - Hata Matrisi\nGenel Doğruluk Skoru: %{accuracy * 100:.2f}\n', 
              fontsize=16, fontweight='bold', color='navy')
    
    plt.ylabel('Gerçek Duygu (Veri Setindeki)', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen Duygu (Modelin Bulduğu)', fontsize=12, fontweight='bold')

    # Grafiği ekranda göster
    plt.tight_layout()
    plt.show()

elif len(X) == 0 and os.path.exists(DATA_PATH):
    print("\nHATA: Klasör bulundu ancak .wav dosyaları işlenemedi.")