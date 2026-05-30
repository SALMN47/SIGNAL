import os
import sys

print("\n" + "="*60)
print(">>> SİSTEM BAŞLATILIYOR... Kütüphaneler yükleniyor, lütfen bekleyin. <<<")
print("="*60 + "\n")

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    """Sese algılanması zor, hafif bir beyaz gürültü ekler."""
    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
    data_noise = data + noise_amp * np.random.normal(size=data.shape[0])
    return data_noise

# ─── 2. GELİŞMİŞ ÖZNİTELİK ÇIKARIMI ───────────
def extract_features(audio, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        rms = librosa.feature.rms(y=audio)
        zcr = librosa.feature.zero_crossing_rate(audio)
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

print("=" * 60)
print("  PHASE 3 - PARAMETRE OPTİMİZASYONU BAŞLADI")
print("=" * 60)

X = []
y = []
file_list = []

# Sadece geçerli .wav dosyalarının listesini oluştur
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

print(f"\n>>> Toplam {len(file_list)} dosya bulundu. (Orijinal + Gürültü) uygulanıyor...")

# tqdm ile ilerleme çubuğu
for path, label in tqdm(file_list, desc="Öznitelik Çıkarımı"):
    try:
        # Sesi bir kere yükle
        audio, sample_rate = librosa.load(path, res_type='kaiser_fast')
        
        # 1. ORİJİNAL SESİ İŞLE VE EKLE
        features_original = extract_features(audio, sample_rate)
        if features_original is not None:
            X.append(features_original)
            y.append(label)
        
        # 2. GÜRÜLTÜLÜ SESİ İŞLE VE EKLE
        audio_noise = add_noise(audio)
        features_noise = extract_features(audio_noise, sample_rate)
        if features_noise is not None:
            X.append(features_noise)
            y.append(label)
    except Exception:
        continue

# ─── 3. MODEL EĞİTİMİ (AGRESİF PARAMETRELER) ──────────────────────────────
if len(X) > 0:
    X = np.array(X)
    y = np.array(y)

    print(f"\n>>> Toplam {len(X)} veri noktası elde edildi. Veri bölünüyor...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(">>> Modeller hiperparametre optimizasyonu ile eğitiliyor, lütfen bekleyin...")
    
    # PARAMETRELER AGRESİF HALE GETİRİLDİ
    rf_model = RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1)
    svm_model = SVC(kernel='rbf', C=25, gamma='scale', probability=True, random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(256, 256, 128), max_iter=700, alpha=0.001, random_state=42)
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('svm', svm_model), ('mlp', mlp_model)],
        voting='soft'
    )
    
    ensemble_model.fit(X_train_scaled, y_train)

    y_pred = ensemble_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    labels_sorted = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print("\n" + "=" * 60)
    print(f"  PHASE 3 NİHAİ DOĞRULUK SKORU: %{accuracy * 100:.2f}")
    print("=" * 60)
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, labels=labels_sorted))
    print("\n>>> İşlem tamamlandı! Ekrana grafik açılıyor...")

    # GÖRSELLEŞTİRME
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=labels_sorted, yticklabels=labels_sorted, 
                annot_kws={"size": 14})

    plt.title(f'Phase 3: Hyperparameter Optimization Hata Matrisi\nNihai Doğruluk Skoru: %{accuracy * 100:.2f}\n', 
              fontsize=16, fontweight='bold', color='darkorange')
    plt.ylabel('Gerçek Duygu', fontsize=12, fontweight='bold')
    plt.xlabel('Tahmin Edilen Duygu', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()