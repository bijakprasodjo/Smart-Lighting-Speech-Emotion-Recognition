import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import librosa
import numpy as np
import time
import glob
from preprocessing import audio_to_features, SR
from inference import SERInference
from sonoff_controller import SonoffController

MODEL_PATH = os.getenv(
    'SER_MODEL_PATH',
    '/home/bijakprasodjo/TA_SER/models/model_C_v2_int8.tflite'
)

RAVDESS_DIR = os.getenv(
    'RAVDESS_DIR',
    '/home/bijakprasodjo/nb4_integration/RAVDESS'
)

EMOTION_CODE = {
    'neutral': '01',
    'happy'  : '03',
    'sad'    : '04',
    'angry'  : '05',
}

EMOTION_ICON = {
    'angry'  : '😠',
    'happy'  : '😊',
    'neutral': '😐',
    'sad'    : '😢'
}

def get_files_by_emotion(emotion, n_per_actor=2, n_actors=5):
   
    code  = EMOTION_CODE[emotion]
    files = []

    for actor_num in range(1, n_actors + 1):
        actor_dir = os.path.join(
            RAVDESS_DIR, f'Actor_{actor_num:02d}'
        )
        
        pattern = os.path.join(
            actor_dir, f'03-01-{code}-*.wav'
        )
        matches = sorted(glob.glob(pattern))

        files.extend(matches[:n_per_actor])

    return files

def run_comprehensive_test(n_per_actor=2, n_actors=5):

    ser    = SERInference(MODEL_PATH)
    sonoff = SonoffController()

    total_files = n_per_actor * n_actors * 4
    print("\n" + "="*62)
    print(" COMPREHENSIVE TEST — RAVDESS MULTI-ACTOR MULTI-FILE")
    print("="*62)
    print(f"   Dataset : RAVDESS ({n_actors} aktor × "
          f"{n_per_actor} file x 4 emosi = {total_files} file)")
    print(f"   Model   : {MODEL_PATH}")
    print("="*62)

    all_results = []

    for emotion in ['angry', 'happy', 'neutral', 'sad']:
        files = get_files_by_emotion(
            emotion, n_per_actor=n_per_actor, n_actors=n_actors
        )

        print(f"\n{'='*62}")
        print(f"  {EMOTION_ICON[emotion]} {emotion.upper()} "
              f"({len(files)} file dari {n_actors} aktor)")
        print(f"{'='*62}")
        print(f"  {'File':<35} {'Pred':<10} {'Conf':>7} "
              f"{'ms':>5} {'OK':>4}")
        print("  " + "-"*58)

        emotion_results = []

        for fpath in files:
            fname = os.path.basename(fpath)
            
            actor_num = fpath.split('Actor_')[1][:2]

            try:
                y, _     = librosa.load(fpath, sr=SR)
                features = audio_to_features(y, sr=SR)
                result   = ser.predict(features)

                pred  = result['emotion']
                conf  = result['confidence']
                inf_t = result['inference_ms']
                ok    = '✅' if pred == emotion else '❌'

                emotion_results.append({
                    'file'        : fname,
                    'actor'       : actor_num,
                    'true'        : emotion,
                    'predicted'   : pred,
                    'confidence'  : conf,
                    'correct'     : pred == emotion,
                    'inference_ms': inf_t
                })

                icon = EMOTION_ICON.get(pred, '?')
                print(f"  Actor{actor_num} {fname:<28} "
                      f"{icon}{pred:<9} "
                      f"{conf*100:>6.1f}% "
                      f"{inf_t:>4.1f}ms {ok}")

            except Exception as e:
                print(f"  {fname:<35} ERROR: {e}")

        # Summary per emosi
        if emotion_results:
            correct  = sum(1 for r in emotion_results if r['correct'])
            total    = len(emotion_results)
            avg_conf = np.mean([r['confidence'] for r in emotion_results])
            avg_inf  = np.mean([r['inference_ms'] for r in emotion_results])

            # Distribusi prediksi
            pred_dist = {}
            for r in emotion_results:
                p = r['predicted']
                pred_dist[p] = pred_dist.get(p, 0) + 1

            print(f"\n  Summary {emotion.upper()}:")
            print(f"     Akurasi      : {correct}/{total} "
                  f"({correct/total*100:.0f}%)")
            print(f"     Avg confidence: {avg_conf*100:.1f}%")
            print(f"     Avg inference : {avg_inf:.2f}ms")
            print(f"     Distribusi prediksi:")
            for pred, count in sorted(pred_dist.items(),
                                      key=lambda x: -x[1]):
                bar  = '█' * count
                icon = EMOTION_ICON.get(pred, '')
                print(f"       {icon}{pred:<10}: {count:>2}x {bar}")

        all_results.extend(emotion_results)

        # Update lampu sesuai emosi ini
        print(f"\n Lampu → {EMOTION_ICON[emotion]} {emotion}...")
        ok_lamp, ms_lamp, cfg = sonoff.set_lighting_by_emotion(emotion)
        if ok_lamp:
            print(f" {cfg['description']} ({ms_lamp:.0f}ms)")
        time.sleep(3)  # jeda 3 detik biar keliatan lampu berubah

    total_correct = sum(1 for r in all_results if r['correct'])
    total_all     = len(all_results)
    overall_acc   = total_correct / total_all if total_all > 0 else 0
    avg_inf_all   = np.mean([r['inference_ms'] for r in all_results])

    print("\n" + "="*62)
    print(" SUMMARY FINAL")
    print("="*62)

    print(f"\n  {'Emosi':<12} {'Benar':>6} {'Total':>6} "
          f"{'Acc':>7} {'Avg Conf':>10}")
    print("  " + "-"*45)

    for emotion in ['angry', 'happy', 'neutral', 'sad']:
        em = [r for r in all_results if r['true'] == emotion]
        if not em:
            continue
        c   = sum(1 for r in em if r['correct'])
        t   = len(em)
        acc = c / t
        avg = np.mean([r['confidence'] for r in em])
        print(f"  {EMOTION_ICON[emotion]}{emotion:<11} "
              f"{c:>6} {t:>6} "
              f"{acc*100:>6.0f}% {avg*100:>9.1f}%")

    print("  " + "-"*45)
    print(f"  {'TOTAL':<12} {total_correct:>6} {total_all:>6} "
        f"{overall_acc*100:>6.1f}%")

    # Confusion Matrix
    print(f"\n CONFUSION MATRIX (baris=true, kolom=predicted):")
    emotions = ['angry', 'happy', 'neutral', 'sad']
    header   = f"  {'':12}"
    for e in emotions:
        header += f" {e[:7]:>8}"
    print(header)
    print("  " + "-"*50)

    for true_e in emotions:
        row = f"  {EMOTION_ICON[true_e]}{true_e:<11}"
        for pred_e in emotions:
            em = [r for r in all_results if r['true'] == true_e]
            count = sum(1 for r in em if r['predicted'] == pred_e)
            marker = '█' if true_e == pred_e else ' '
            row += f" {count:>7}{marker}"
        print(row)

    print("\n  (█ = diagonal = prediksi BENAR)")

    print(f"\n Avg inference : {avg_inf_all:.2f} ms per file")
    print(f"  File test ini: {overall_acc*100:.1f}%")
    print("="*62)

if __name__ == '__main__':
    run_comprehensive_test(n_per_actor=2, n_actors=5)