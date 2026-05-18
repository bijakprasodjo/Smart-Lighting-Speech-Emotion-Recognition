import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import json
import psutil
from datetime import datetime
from collections import deque
from collections import Counter

from audio_stream import AudioStream
from preprocessing import audio_to_features, SR, is_speech_like
from inference import SERInference, LIGHTING_RULES
from mqtt_client import MQTTPublisher
from sonoff_controller import SonoffController

CONFIG = {
    'model_path'         : os.getenv('SER_MODEL_PATH', '/home/bijakprasodjo/TA_SER/models/model_C_v2_int8.tflite'),
    'mic_sr'             : 44100,
    'mic_device'         : 1,
    'channels'           : 2,
    'window_duration'    : 3.0,    
    'slide_interval'     : 0.5,    
    'conf_threshold'     : 0.6,
    'vad_threshold'      : 0.05,
    'speech_check'       : True,
    'flatness_threshold' : 0.3,
    'smooth_window'      : 6,
    'mqtt_broker'        : 'localhost',
    'mqtt_port'          : 1883,
    'mqtt_qos'           : 1,
    'silence_timeout'    : 30,
    'ambient_timeout'    : 60,
    'log_path'           : os.getenv('LOG_PATH', '/home/bijakprasodjo/TA_SER/logs/stream_log.jsonl'),
}

def is_voice_detected(audio, threshold=0.04):
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms >= threshold, round(rms, 5)

class EmotionSmoother:
   
    def __init__(self, window=6, majority_ratio=0.60, min_count=4):
        self.window         = window
        self.majority_ratio = majority_ratio  
        self.min_count      = min_count       
        self.history        = deque(maxlen=window)

    def update(self, emotion):
        self.history.append(emotion)

        # Belum cukup data
        if len(self.history) < self.window:
            return None

        # Hitung frekuensi tiap emosi
        counts = Counter(self.history)
        top_emotion, top_count = counts.most_common(1)[0]

        # Trigger kalau memenuhi threshold
        ratio = top_count / self.window
        if ratio >= self.majority_ratio and top_count >= self.min_count:
            return top_emotion

        return None

    def reset(self):
        self.history.clear()

    @property
    def status(self):
        if not self.history:
            return 'buf:0'
        from collections import Counter
        counts  = Counter(self.history)
        top_e, top_c = counts.most_common(1)[0]
        return f'{top_e[:3]}:{top_c}/{self.window}'

def get_system_stats():
    return {
        'cpu_pct': psutil.cpu_percent(interval=None),
        'ram_pct': psutil.virtual_memory().percent,
    }

def log_result(log_path, record):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(record) + '\n')

def reset_to_neutral(sonoff, mqtt_pub, config,
                     counter, ts, trigger_name):
    ok, ms, _ = sonoff.set_lighting_by_emotion('neutral')
    if ok:
        print(f"\n{'⏰' if trigger_name=='silence' else '🌫️ '} "
              f"{trigger_name.capitalize()} timeout "
              f"→ Reset ke NEUTRAL ({ms:.0f}ms)")
        mqtt_pub.publish_result(
            {
                'emotion'      : 'neutral',
                'confidence'   : 1.0,
                'probabilities': {
                    'angry':0,'happy':0,'neutral':1,'sad':0
                },
                'inference_ms' : 0,
                'lighting'     : LIGHTING_RULES['neutral']
            },
            timestamp=ts, qos=config['mqtt_qos']
        )
        log_result(config['log_path'], {
            'counter'       : counter,
            'timestamp'     : ts,
            'datetime'      : datetime.fromtimestamp(ts).isoformat(),
            'trigger'       : f'{trigger_name}_timeout',
            'emotion'       : 'neutral',
            'stable_emotion': 'neutral',
            'lamp_ms'       : ms,
            'current_lamp'  : 'neutral',
        })
    return ok, ms

def run_stream_pipeline(config=CONFIG):
    print("\n" + "="*65)
    print(" SER STREAMING Pipeline — Smart Lighting")
    print("="*65)
    print(f"   Model          : {config['model_path']}")
    print(f"   Window         : {config['window_duration']}s")
    print(f"   Slide interval : {config['slide_interval']}s")
    print(f"   VAD thresh     : {config['vad_threshold']} RMS")
    print(f"   Speech check   : {'ON' if config['speech_check'] else 'OFF'}")
    print(f"   Smooth         : {config['smooth_window']}x berturut-turut")
    print(f"   Silence timeout: {config['silence_timeout']}s → neutral")
    print(f"   Ambient timeout: {config['ambient_timeout']}s → neutral")
    print("="*65 + "\n")

    stream = AudioStream(
        mic_sr   = config['mic_sr'],
        target_sr= SR,
        device   = config['mic_device'],
        channels = config['channels'],
    )

    ser = SERInference(config['model_path'])

    mqtt_pub = MQTTPublisher(
        broker=config['mqtt_broker'],
        port  =config['mqtt_port']
    )
    if not mqtt_pub.connect():
        print(" Gagal connect ke MQTT broker!")
        return

    sonoff   = SonoffController()
    smoother = EmotionSmoother(window=config['smooth_window'])

    stream.start()

    print(f" Mengisi buffer {config['window_duration']}s...")
    time.sleep(config['window_duration'])
    print(" Buffer siap! Mulai inference...\n")

    print(f"{'#':<5} {'Status':<14} {'Emotion':<12} "
          f"{'Conf':>6} {'RMS':>7} {'Lamp':>12}")
    print("-"*62)

    counter             = 0
    lamp_counter        = 0
    total_inf           = []
    total_lamp          = []
    current_lamp        = None
    silence_duration    = 0.0
    no_emotion_duration = 0.0

    try:
        while True:
            counter += 1
            ts         = time.time()
            t_start    = time.perf_counter()

            audio = stream.get_window()
            
            voice_detected, rms = is_voice_detected(
                audio, config['vad_threshold']
            )

            if not voice_detected:
                smoother.reset()
                no_emotion_duration  = 0
                silence_duration    += config['slide_interval']

                if (silence_duration >= config['silence_timeout']
                        and current_lamp not in (None, 'neutral')):
                    ok, _ = reset_to_neutral(
                        sonoff, mqtt_pub, config,
                        counter, ts, 'silence'
                    )
                    if ok:
                        current_lamp     = 'neutral'
                        silence_duration = 0

                print(f"{counter:<5} {'[silence]':<14} "
                      f"{'—':<12} {'—':>6} {rms:>7.5f} "
                      f"[diam {silence_duration:.0f}s]")

                # Tunggu slide_interval sebelum ambil window berikutnya
                elapsed = (time.perf_counter() - t_start) * 1000
                sleep_t = max(0, config['slide_interval']
                              - elapsed/1000)
                time.sleep(sleep_t)
                continue

            silence_duration = 0

            if config['speech_check']:
                speech_like, flatness = is_speech_like(
                    audio, config['flatness_threshold']
                )
                if not speech_like:
                    no_emotion_duration += config['slide_interval']

                    if (no_emotion_duration >= config['ambient_timeout']
                            and current_lamp not in (None, 'neutral')):
                        ok, _ = reset_to_neutral(
                            sonoff, mqtt_pub, config,
                            counter, ts, 'ambient'
                        )
                        if ok:
                            current_lamp        = 'neutral'
                            no_emotion_duration = 0

                    print(f"{counter:<5} {'[non-speech]':<14} "
                          f"{'—':<12} {'—':>6} {rms:>7.5f} "
                          f"[flat={flatness:.2f}]")

                    elapsed = (time.perf_counter() - t_start) * 1000
                    sleep_t = max(0, config['slide_interval']
                                  - elapsed/1000)
                    time.sleep(sleep_t)
                    continue

            features   = audio_to_features(audio, sr=SR)
            result     = ser.predict(features)
            total_inf.append(result['inference_ms'])
            emotion    = result['emotion']
            confidence = result['confidence']

            if confidence < config['conf_threshold']:
                no_emotion_duration += config['slide_interval']

                if (no_emotion_duration >= config['ambient_timeout']
                        and current_lamp not in (None, 'neutral')):
                    ok, _ = reset_to_neutral(
                        sonoff, mqtt_pub, config,
                        counter, ts, 'ambient'
                    )
                    if ok:
                        current_lamp        = 'neutral'
                        no_emotion_duration = 0

                print(f"{counter:<5} {'[low conf]':<14} "
                      f"{emotion:<12} {confidence:>5.1%} "
                      f"{rms:>7.5f} "
                      f"[amb {no_emotion_duration:.0f}s]")

                elapsed = (time.perf_counter() - t_start) * 1000
                sleep_t = max(0, config['slide_interval']
                              - elapsed/1000)
                time.sleep(sleep_t)
                continue

            no_emotion_duration = 0

            stable_emotion = smoother.update(emotion)

            lamp_ms     = 0
            lamp_status = '—'
            mqtt_ok     = False
            mqtt_ms     = 0

            if stable_emotion and stable_emotion != current_lamp:
                lamp_ok, lamp_ms, _ = sonoff.set_lighting_by_emotion(stable_emotion)

                if lamp_ok:
                    current_lamp = stable_emotion
                    lamp_counter += 1
                    total_lamp.append(lamp_ms)
                    lamp_status = f'{lamp_ms:.0f}ms'

                    # Publish stable emotion, not raw frame-level prediction
                    stable_result = dict(result)
                    stable_result['emotion'] = stable_emotion
                    stable_result['confidence'] = float(
                        result['probabilities'].get(stable_emotion, result['confidence'])
                    )
                    stable_result['lighting'] = LIGHTING_RULES[stable_emotion]

                    mqtt_ok, mqtt_ms = mqtt_pub.publish_result(
                        stable_result,
                        timestamp=ts,
                        qos=config['mqtt_qos']
                    )

            elif stable_emotion == current_lamp:
                lamp_status = 'same'
            else:
                lamp_status = smoother.status

            stats = get_system_stats()
            log_result(config['log_path'], {
                'counter'       : counter,
                'timestamp'     : ts,
                'datetime'      : datetime.fromtimestamp(ts).isoformat(),
                'emotion'       : emotion,
                'stable_emotion': stable_emotion,
                'confidence'    : confidence,
                'rms'           : rms,
                'inference_ms'  : result['inference_ms'],
                'lamp_ms'       : lamp_ms,
                'current_lamp'   : current_lamp,
                'mqtt_ok'        : mqtt_ok,
                'mqtt_publish_ms': mqtt_ms,
                'cpu_pct'        : stats['cpu_pct'],
                'ram_pct'        : stats['ram_pct'],
            })

            emotion_icon = {
                'angry':'😠','happy':'😊',
                'neutral':'😐','sad':'😢'
            }.get(emotion, '?')

            print(f"{counter:<5} {'[voice]':<14} "
                  f"{emotion_icon}{emotion:<11} "
                  f"{confidence:>5.1%} "
                  f"{rms:>7.5f} "
                  f"{lamp_status:>12}")

            elapsed = (time.perf_counter() - t_start) * 1000
            sleep_t = max(0, config['slide_interval'] - elapsed/1000)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n Pipeline dihentikan (Ctrl+C)\n")

    finally:
        stream.stop()
        mqtt_pub.disconnect()

        if total_inf:
            print("="*55)
            print(" SUMMARY SESSION — STREAMING")
            print("="*55)
            print(f"   Total inference : {counter}")
            print(f"   Lamp updates    : {lamp_counter}")
            print(f"   Inference mean  : {np.mean(total_inf):.2f} ms")
            print(f"   Inference max   : {np.max(total_inf):.2f} ms")
            if total_lamp:
                print(f"   Lamp resp mean  : {np.mean(total_lamp):.2f} ms")
            print(f"   Slide interval  : {config['slide_interval']}s")
            print(f"   Log             : {config['log_path']}")
            print("="*55)

if __name__ == '__main__':
    run_stream_pipeline()