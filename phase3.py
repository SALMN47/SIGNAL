import os
import sys

print("\n" + "="*60)
print(">>> SİSTEM BAŞLATILIYOR... Zirve Skor İçin Ultimate Modifikasyonlar Yükleniyor. <<<")
print("="*60 + "\n")

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

print(">>> Kütüphaneler başarıyla yüklendi. İşlem hazırlıkları yapılıyor...\n")

# ─── 1. VERİ ÇOĞALTMA ─────────────────────────────
def add_noise(data):
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data_noise = data + noise_amp * np.random.normal(size=data.shape[0])
    return data_noise

# ─── 2. ULTIMATE ÖZNİTELİK ÇIKARIMI (DELTA EKLENDİ) ───────────
def extract_features(audio, sample_rate):
    try:
        # MFCC ve Türevleri (Sesin hızı ve ivmesi)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        rms = librosa.feature.rms(y=audio)
        zcr = librosa.feature.zero_crossing_rate(audio)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        
        features = np.hstack([
            np.mean(mfccs.T, axis=0), np.std(mfccs.T, axis=0),
            np.mean(delta_mfccs.T, axis=0), np.std(delta_mfccs.T, axis=0),    # YENİ
            np.mean(delta2_mfccs.T, axis=0), np.std(delta2_mfccs.T, axis=0),  # YENİ
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

print("=" * 60)
print("  PHASE 3 - ULTIMATE LOCKDOWN BAŞLADI")
print("  Leaderboard Hedefi: %94+ Doğruluk Skoru")
print("=" * 60)

X = []
y = []
file_list = []

if os.path.exists(DATA_PATH):
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".wav"):
                filename_lower = file.lower()
                for key, label in emotion_mapping.items():
                    if key in filename_lower:
                        file_list.append((os.path.join(root, file), label))
                        break
else:
    print(f"HATA: '{DATA_PATH}' bulunamadı.")
    sys.exit()

print(f"\n>>> Toplam {len(file_list)} dosya bulundu. Çift işlem (Orijinal + Gürültü) uygulanıyor...")

for path, label in tqdm(file_list, desc="Öznitelik Çıkarımı"):
    try:
        audio, sample_rate = librosa.load(path, res_type='kaiser_fast')
        
        # 1. ORİJİNAL
        features_original = extract_features(audio, sample_rate)
        if features_original is not None:
            X.append(features_original)
            y.append(label)
        
        # 2. GÜRÜLTÜLÜ
        audio_noise = add_noise(audio)
        features_noise = extract_features(audio_noise, sample_rate)
        if features_noise is not None:
            X.append(features_noise)
            y.append(label)
    except Exception:
        continue

# ─── 3. ULTIMATE MODEL EĞİTİMİ (STACKING CLASSIFIER) ──────────────────────────────
if len(X) > 0:
    X = np.array(X)
    y = np.array(y)

    print(f"\n>>> Toplam {len(X)} veri noktası ve {X.shape[1]} öznitelik elde edildi. Veri bölünüyor...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(">>> Stacking Ensemble Modeli (4 Güçlü Algoritma) eğitiliyor, lütfen bekleyin...")
    
    # EN GÜÇLÜ 4 ALGORİTMA
    rf_model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    svm_model = SVC(kernel='rbf', C=30, gamma='scale', probability=True, random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=800, alpha=0.0005, random_state=42)
    hgb_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
    
    # STACKING CLASSIFIER (NİHAİ SİLAH)
    stacking_model = StackingClassifier(
        estimators=[('rf', rf_model), ('svm', svm_model), ('mlp', mlp_model), ('hgb', hgb_model)],
        final_estimator=LogisticRegression(C=2.0, max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train_scaled, y_train)

    y_pred = stacking_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    labels_sorted = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print("\n" + "=" * 60)
    print(f"  PHASE 3 ULTIMATE DOĞRULUK SKORU: %{accuracy * 100:.2f}")
    print("=" * 60)
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, labels=labels_sorted))
    print("\n>>> İşlem tamamlandı! Ekrana grafik açılıyor...")

    # GÖRSELLEŞTİRME
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_sorted, yticklabels=labels_sorted, 
                annot_kws={"size": 14})

    plt.title(f'Phase 3: Ultimate Stacking Ensemble\nNihai Doğruluk Skoru: %{accuracy * 100:.2f}\n', 
              fontsize=16, fontweight='bold', color='darkblue')
    plt.ylabel('Gerçek Duygu', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen Duygu', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
