import os
import requests
import json
import time

SONOFF_IP   = os.getenv('SONOFF_IP', '192.168.0.100')
SONOFF_PORT = int(os.getenv('SONOFF_PORT', '8081'))
DEVICE_ID   = os.getenv('SONOFF_DEVICE_ID', 'YOUR_DEVICE_ID')

EMOTION_LIGHTING = {
    'angry': {
        'ltype'      : 'color',
        'color'      : {'r': 0, 'g': 100, 'b': 255, 'br': 40},
        'description': 'Cool blue — menenangkan emosi marah'
    },
    'happy': {
        'ltype'      : 'white',
        'white'      : {'br': 80, 'ct': 100},
        'description': 'Neutral white — mempertahankan mood positif'
    },
    'neutral': {
        'ltype'      : 'white',
        'white'      : {'br': 70, 'ct': 50},
        'description': 'Warm white — mendukung fokus kerja'
    },
    'sad': {
        'ltype'      : 'color',
        'color'      : {'r': 255, 'g': 200, 'b': 0, 'br': 90},
        'description': 'Warm yellow — meningkatkan mood sedih'
    },
}

TRANSITION_STEPS = 5    # jumlah langkah transisi
TRANSITION_DELAY = 0.3  # detik antar step (total ~1.5s)

def interpolate_color(c_from, c_to, t):
    """Linear interpolasi antara 2 color dict, t = 0.0..1.0"""
    return {
        k: int(c_from[k] + (c_to[k] - c_from[k]) * t)
        for k in c_to
    }

def interpolate_white(w_from, w_to, t):
    """Linear interpolasi antara 2 white dict, t = 0.0..1.0"""
    return {
        k: int(w_from[k] + (w_to[k] - w_from[k]) * t)
        for k in w_to
    }

# Default "neutral" sebagai starting point kalau state awal tidak diketahui
DEFAULT_STATE = {
    'ltype': 'white',
    'white': {'br': 70, 'ct': 50}
}

class SonoffController:
    def __init__(self, ip=SONOFF_IP, port=SONOFF_PORT,
                 device_id=DEVICE_ID):
        self.ip        = ip
        self.port      = port
        self.device_id = device_id
        self.url       = f'http://{ip}:{port}/zeroconf/dimmable'
        self.timeout   = 3
        self.last_emotion = None
        # Track current state untuk gradual transition
        self._current_state = DEFAULT_STATE.copy()

    def _send_command(self, data):
        payload = {
            'deviceid': self.device_id,
            'data'    : data
        }
        try:
            start = time.perf_counter()
            resp  = requests.post(
                self.url, json=payload, timeout=self.timeout
            )
            elapsed = (time.perf_counter() - start) * 1000
            result  = resp.json()
            success = (result.get('error', -1) == 0)
            return success, round(elapsed, 2)
        except requests.exceptions.Timeout:
            return False, -1
        except requests.exceptions.ConnectionError:
            return False, -1
        except Exception as e:
            print(f"⚠️  Sonoff error: {e}")
            return False, -1

    def _get_state_dict(self, config):
        """Ambil dict parameter dari config emosi."""
        if config['ltype'] == 'color':
            return {'ltype': 'color', 'color': config['color'].copy()}
        else:
            return {'ltype': 'white', 'white': config['white'].copy()}

    def set_lighting_gradual(self, emotion,
                              steps=TRANSITION_STEPS,
                              delay=TRANSITION_DELAY):
        """
        Set lampu dengan gradual transition.
        Referensi: Kompier et al. (2021) — hindari abrupt transition.

        Returns: (success, total_response_ms, config)
        """
        if emotion not in EMOTION_LIGHTING:
            return False, -1, None

        target_config = EMOTION_LIGHTING[emotion]
        target_state  = self._get_state_dict(target_config)
        from_state    = self._current_state

        total_ms  = 0
        last_ok   = False

        # Kalau tipe sama (color→color atau white→white) → interpolasi
        # Kalau beda tipe → langsung set (tidak bisa interpolasi)
        same_type = (from_state['ltype'] == target_state['ltype'])

        for step in range(1, steps + 1):
            t = step / steps  # 0.2, 0.4, 0.6, 0.8, 1.0

            if same_type:
                if target_state['ltype'] == 'color':
                    interp = interpolate_color(
                        from_state.get('color', target_state['color']),
                        target_state['color'], t
                    )
                    data = {'ltype': 'color', 'color': interp}
                else:
                    interp = interpolate_white(
                        from_state.get('white', target_state['white']),
                        target_state['white'], t
                    )
                    data = {'ltype': 'white', 'white': interp}
            else:
                # Beda tipe → langsung ke target di step pertama
                data = target_state
                t    = 1.0  # skip langsung ke akhir

            ok, ms = self._send_command(data)
            total_ms += ms if ms > 0 else 0
            last_ok   = ok

            if not ok:
                break

            if step < steps:
                time.sleep(delay)

            # Kalau beda tipe → cukup 1 step
            if not same_type:
                break

        if last_ok:
            self._current_state = target_state
            self.last_emotion   = emotion

        return last_ok, round(total_ms, 2), target_config

    def set_lighting_by_emotion(self, emotion):
        """
        Wrapper utama — pakai gradual transition.
        Signature sama dengan versi lama agar kompatibel.
        """
        return self.set_lighting_gradual(emotion)

    def turn_off(self):
        try:
            requests.post(
                f'http://{self.ip}:{self.port}/zeroconf/switch',
                json={'deviceid': self.device_id,
                      'data': {'switch': 'off'}},
                timeout=self.timeout
            )
        except:
            pass

    def test_all_emotions(self):
        print("\n Test semua mapping emosi → lampu (gradual)\n")
        print(f"{'Emosi':<10} {'Status':>8} {'Latency':>10} Deskripsi")
        print("-" * 60)
        for emotion in ['angry', 'happy', 'neutral', 'sad']:
            ok, ms, cfg = self.set_lighting_by_emotion(emotion)
            status = 'OK' if ok else 'FAIL'
            desc   = cfg['description'] if cfg else '-'
            print(f"{emotion:<10} {status:>8} {ms:>8.1f}ms  {desc}")
            time.sleep(3)
        print("\n Test selesai!")

if __name__ == '__main__':
    sonoff = SonoffController()
    sonoff.test_all_emotions()