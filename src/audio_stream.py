import numpy as np
import sounddevice as sd
import librosa
import threading
from collections import deque

# ============================================================
# KONFIGURASI
# ============================================================
MIC_SR     = 44100   # sample rate native fifine mic
TARGET_SR  = 16000   # sample rate target model
TARGET_DUR = 3.0     # detik window
TARGET_LEN = int(TARGET_SR * TARGET_DUR)  # 48000 sampel

# Buffer size = 3 detik audio di target SR
# Diisi oleh callback mic secara continuous
BUFFER_DURATION = TARGET_DUR   # 3 detik
BUFFER_SAMPLES  = int(TARGET_SR * BUFFER_DURATION)  # 48000

class AudioStream:
    """
    Continuous audio capture dengan sliding window.

    Cara kerja:
    - sounddevice callback dipanggil tiap ~20ms oleh OS
    - Setiap chunk audio di-resample ke 16kHz
    - Di-push ke circular buffer (deque)
    - get_window() ambil 3 detik terakhir kapanpun dibutuhkan
    """

    def __init__(self, mic_sr=MIC_SR, target_sr=TARGET_SR,
                 device=1, channels=2,
                 buffer_duration=BUFFER_DURATION,
                 chunk_duration=0.1):
        self.mic_sr           = mic_sr
        self.target_sr        = target_sr
        self.device           = device
        self.channels         = channels
        self.chunk_duration   = chunk_duration  # 100ms per chunk
        self.chunk_samples_mic = int(mic_sr * chunk_duration)

        # Circular buffer — simpan audio dalam target SR
        buffer_samples = int(target_sr * buffer_duration)
        self.buffer    = deque(maxlen=buffer_samples)
        self.lock      = threading.Lock()

        self._stream   = None
        self._running  = False

        # Pre-fill buffer dengan silence
        with self.lock:
            self.buffer.extend(np.zeros(buffer_samples))

        print(f"🎙️  AudioStream init:")
        print(f"   Mic SR     : {mic_sr} Hz")
        print(f"   Target SR  : {target_sr} Hz")
        print(f"   Chunk      : {chunk_duration*1000:.0f}ms")
        print(f"   Buffer     : {buffer_duration}s = {buffer_samples} sampel")

    def _callback(self, indata, frames, time_info, status):
        """
        Dipanggil sounddevice tiap chunk_duration ms.
        Push audio ke buffer.
        """
        if status:
            pass  # ignore overflow/underflow warning

        # Convert stereo → mono
        if indata.ndim > 1:
            chunk = np.mean(indata, axis=1).astype(np.float32)
        else:
            chunk = indata.flatten().astype(np.float32)

        # Resample dari mic_sr ke target_sr
        if self.mic_sr != self.target_sr:
            chunk = librosa.resample(
                chunk,
                orig_sr=self.mic_sr,
                target_sr=self.target_sr
            )

        # Push ke buffer (thread-safe)
        with self.lock:
            self.buffer.extend(chunk)

    def start(self):
        """Mulai stream mic."""
        self._running = True
        self._stream  = sd.InputStream(
            samplerate  = self.mic_sr,
            device      = self.device,
            channels    = self.channels,
            dtype       = 'float32',
            blocksize   = self.chunk_samples_mic,
            callback    = self._callback
        )
        self._stream.start()
        print("✅ AudioStream started!")

    def stop(self):
        """Stop stream mic."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._running = False
        print("⏹️  AudioStream stopped!")

    def get_window(self):
        """
        Ambil 3 detik terakhir dari buffer.
        Returns: numpy array shape (48000,) float32
        """
        with self.lock:
            return np.array(self.buffer, dtype=np.float32)

    def get_rms(self):
        """RMS dari window saat ini untuk VAD."""
        window = self.get_window()
        return float(np.sqrt(np.mean(window ** 2)))

    @property
    def is_running(self):
        return self._running