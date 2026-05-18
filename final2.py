import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- YENİ: VERİ ÇOĞALTMA (AUGMENTATION) FONKSİYONU ---
def add_noise(data):
    """Sese algılanması zor, hafif bir beyaz gürültü ekler."""
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data_noise = data + noise_amp * np.random.normal(size=data.shape[0])
    return data_noise

# --- YENİ: SPEKTRAL ÖZNİTELİKLERLE ZENGİNLEŞTİRİLMİŞ ÇIKARIM ---
def extract_features(audio, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        rms = librosa.feature.rms(y=audio)
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Yeni Eklenenler
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        
        features = np.hstack([
            np.mean(mfccs.T, axis=0), np.std(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0), np.std(chroma.T, axis=0),
            np.mean(mel.T, axis=0), np.std(mel.T, axis=0),
            np.mean(rms.T, axis=0), np.std(rms.T, axis=0),
            np.mean(zcr.T, axis=0), np.std(zcr.T, axis=0),
            np.mean(centroid.T, axis=0), np.std(centroid.T, axis=0),
            np.mean(bandwidth.T, axis=0), np.std(bandwidth.T, axis=0),
            np.mean(rolloff.T, axis=0), np.std(rolloff.T, axis=0)
        ])
        return features
    except Exception as e:
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

print("FAZ 2 KİLİDİ AÇILDI: Veri çoğaltma (Augmentation) ve Spektral Analiz devrede.")
print("Sesler iki kez işleniyor (Orijinal + Gürültülü). Bu işlem bilgisayarınızı biraz yorabilir, lütfen bekleyin...\n")

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
                    # Sesi bir kere yükle
                    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                    
                    # 1. ORİJİNAL SESİ İŞLE VE EKLE
                    features_original = extract_features(audio, sample_rate)
                    if features_original is not None:
                        X.append(features_original)
                        y.append(target_emotion)
                    
                    # 2. VERİ ÇOĞALTMA (GÜRÜLTÜLÜ SES) İŞLE VE EKLE
                    audio_noise = add_noise(audio)
                    features_noise = extract_features(audio_noise, sample_rate)
                    if features_noise is not None:
                        X.append(features_noise)
                        y.append(target_emotion)
else:
    print(f"HATA: '{DATA_PATH}' bulunamadı.")

# 3. YENİ MODEL: ENSEMBLE (OYLAMA) SİSTEMİ
if len(X) > 0:
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Kolektif Yapay Zeka Eğitiliyor (Random Forest + SVM + Sinir Ağı)...")
    
    # 3 Farklı modeli tanımlıyoruz
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, alpha=0.01, random_state=42)
    
    # Modelleri "Soft Voting" (Olasılıkları toplayarak karar verme) ile birleştiriyoruz
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('svm', svm_model), ('mlp', mlp_model)],
        voting='soft'
    )
    
    ensemble_model.fit(X_train_scaled, y_train)

    y_pred = ensemble_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    labels = ensemble_model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print(f"\nİşlem Tamam! Orijinal + Çoğaltılmış Toplam {len(X)} veri noktası eğitildi.")
    print(f"ULTIMATE Doğruluk Skoru: %{accuracy * 100:.2f}. Ekrana grafik açılıyor...")

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=labels, yticklabels=labels, 
                annot_kws={"size": 14})

    plt.title(f'Faz 2/3: Ensemble Model Hata Matrisi\nYeni Doğruluk Skoru: %{accuracy * 100:.2f}\n', 
              fontsize=16, fontweight='bold', color='darkorange')
    plt.ylabel('Gerçek Duygu', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen Duygu', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()