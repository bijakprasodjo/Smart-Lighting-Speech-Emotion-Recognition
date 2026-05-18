import numpy as np
import librosa
import pickle
import os

SR         = 16000
MIC_SR     = 44100
TARGET_DUR = 3.0
TARGET_LEN = int(SR * TARGET_DUR)  

N_MFCC   = 40
HOP_LEN  = 256
N_FFT    = 512
N_FRAMES = int(np.ceil(TARGET_LEN / HOP_LEN))  

SCALER_DIR = os.getenv(
    'SCALER_DIR',
    '/home/bijakprasodjo/TA_SER/models/scalers'
)
SCALER_PATHS  = {
    'ravdess': os.path.join(SCALER_DIR, 'scaler_ravdess.pkl'),
    'emodb'  : os.path.join(SCALER_DIR, 'scaler_emodb.pkl'),
    'savee'  : os.path.join(SCALER_DIR, 'scaler_savee.pkl'),
    'tess'   : os.path.join(SCALER_DIR, 'scaler_tess.pkl'),
}

_scalers = {}
for src, path in SCALER_PATHS.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            _scalers[src] = pickle.load(f)
    else:
        print(f" Scaler tidak ditemukan: {path}")

DEFAULT_SCALER = 'ravdess'

def compute_spectral_flatness(y, n_fft=N_FFT, hop_length=HOP_LEN):
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )
    return float(np.mean(flatness))

def is_speech_like(y, flatness_threshold=0.3):
    flatness = compute_spectral_flatness(y)
    return flatness < flatness_threshold, flatness

def preprocess_audio(y, sr=SR, target_len=TARGET_LEN):
   
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    if len(y_trimmed) < SR * 0.5:
        y_trimmed = y

    active_rms = np.sqrt(np.mean(y_trimmed ** 2))

    if active_rms > 1e-6:
        target_rms = 0.15
        gain       = target_rms / active_rms
        gain       = np.clip(gain, 0.1, 30.0)
        y_trimmed  = y_trimmed * gain
        y_trimmed  = np.clip(y_trimmed, -1.0, 1.0)

    noise_floor = 0.02
    y_trimmed   = np.where(
        np.abs(y_trimmed) > noise_floor,
        y_trimmed, 0.0
    )

    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_trimmed = y_trimmed / max_val * 0.95

    if len(y_trimmed) < target_len:
        pad_len   = target_len - len(y_trimmed)
        y_trimmed = np.pad(y_trimmed, (0, pad_len), mode='constant')
    else:
        start     = (len(y_trimmed) - target_len) // 2
        y_trimmed = y_trimmed[start: start + target_len]

    return y_trimmed

def extract_mfcc(y, sr=SR, n_mfcc=N_MFCC,
                 hop_length=HOP_LEN, n_fft=N_FFT):
    
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=n_fft
    )
    return mfcc.T.astype(np.float32)  

def apply_scaler(mfcc, scaler_name=DEFAULT_SCALER):
   
    scaler = _scalers.get(scaler_name)

    if scaler is None:
        print(f" Scaler '{scaler_name}' tidak ada, pakai z-score fallback")
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std  = np.std(mfcc, axis=0, keepdims=True)
        std  = np.where(std == 0, 1e-8, std)
        return ((mfcc - mean) / std).astype(np.float32)
    
    n_frames, n_mfcc = mfcc.shape
    mfcc_2d    = mfcc.reshape(-1, n_mfcc)
    mfcc_scaled = scaler.transform(mfcc_2d)
    return mfcc_scaled.reshape(n_frames, n_mfcc).astype(np.float32)

def audio_to_features(y, sr=SR, scaler_name=DEFAULT_SCALER):
    
    y_proc      = preprocess_audio(y, sr=sr)
    mfcc        = extract_mfcc(y_proc)
    mfcc_scaled = apply_scaler(mfcc, scaler_name=scaler_name)

    assert mfcc_scaled.shape == (N_FRAMES, N_MFCC), \
        f"Shape error: got {mfcc_scaled.shape}, expected ({N_FRAMES}, {N_MFCC})"

    return mfcc_scaled[np.newaxis, ...]