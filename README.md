# Smart-Lighting Speech Emotion Recognition

This repository contains the implementation files, trained TensorFlow Lite model, Raspberry Pi inference scripts, MQTT integration scripts, and experiment logs for the paper:

**Lightweight Multi-Branch Dilated 1D CNN for Edge-Based Speech Emotion Recognition with Adaptive Smart Lighting Feedback via MQTT**

The system performs real-time Speech Emotion Recognition (SER) on a Raspberry Pi 4B and uses the detected stable emotion label to trigger adaptive smart-lighting feedback. MQTT is used for local event publication, while the physical Sonoff B05 smart bulb is actuated through the Sonoff local HTTP API.

## Main Contributions

- Lightweight multi-branch dilated 1D CNN for 4-class SER: **angry, happy, neutral, sad**
- Multi-corpus training using **RAVDESS, EMODB, TESS, and SAVEE**
- TensorFlow Lite INT8 deployment on Raspberry Pi 4B
- Real-time streaming inference with overlapping sliding windows and majority voting
- MQTT QoS 1 message delivery ratio evaluation
- Sonoff B05 smart-light actuation through local HTTP commands
- Runtime logs for edge inference, streaming performance, and communication/actuation testing

## Repository Structure

```text
Smart-Lighting-Speech-Emotion-Recognition/
├── README.md
├── requirements.txt
├── docs/
│   └── data_program_availability_evidence_inventory.txt
├── logs/
│   ├── comprehensive_test_output.txt
│   ├── mqtt_mdr_100_messages_output.txt
│   ├── mqtt_mdr_100_messages_output.json
│   └── stream_log.jsonl
├── models/
│   ├── model_C_v2_int8.tflite
│   ├── model_metadata.json
│   └── scalers/
│       ├── scaler_ravdess.pkl
│       ├── scaler_emodb.pkl
│       ├── scaler_savee.pkl
│       └── scaler_tess.pkl
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03a_train_model_A.ipynb
│   ├── 03b_train_model_B.ipynb
│   ├── 03c_train_model_C.ipynb
│   ├── 03c_v2_train_model_C_v2.ipynb
│   ├── 03c_v3_train_model_C_v3.ipynb
│   └── 04_evaluation_and_tflite.ipynb
├── results/
│   ├── evaluation_summary.csv
│   ├── training_curves_A.png
│   ├── training_curves_B.png
│   ├── training_curves_C.png
│   ├── training_curves_C_v2.png
│   └── training_curves_C_v3.png
└── src/
    ├── audio_stream.py
    ├── preprocessing.py
    ├── inference.py
    ├── main_stream.py
    ├── mqtt_client.py
    ├── sonoff_controller.py
    ├── message_delivery_ratio.py
    └── comprehensive_test.py
```

## Datasets

The raw speech datasets are **not redistributed** in this repository because they are third-party datasets with their own licensing terms. Please download the datasets from their official sources:

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song, Zenodo
- **EMODB**: Berlin Database of Emotional Speech, TU Berlin
- **TESS**: Toronto Emotional Speech Set, University of Toronto
- **SAVEE**: Surrey Audio-Visual Expressed Emotion, University of Surrey

After downloading the datasets, adjust the local paths in the dataset preparation notebook or through environment variables when running Raspberry Pi tests.

## Dataset Summary

After label harmonization, only four target classes are retained: **angry, happy, neutral, and sad**.

| Dataset | Angry | Happy | Neutral | Sad | Total |
|---|---:|---:|---:|---:|---:|
| RAVDESS | 192 | 192 | 96 | 192 | 672 |
| EMODB | 127 | 71 | 79 | 62 | 339 |
| TESS | 400 | 400 | 400 | 400 | 1600 |
| SAVEE | 60 | 60 | 120 | 60 | 300 |
| **Total** | **779** | **723** | **695** | **714** | **2911** |

The dataset split is **80/10/10 stratified per source**, resulting in:

- Training set: **2,328 files**
- Validation set: **291 files**
- Test set: **292 files**
- Training set after augmentation: **9,312 samples**

## Feature Extraction

The system uses MFCC features with the following configuration:

| Parameter | Value |
|---|---:|
| Sampling rate | 16,000 Hz |
| Audio duration | 3 seconds |
| MFCC coefficients | 40 |
| FFT size | 512 |
| Hop length | 256 |
| Feature shape | `(188, 40)` |
| Single inference input shape | `(1, 188, 40)` |

Per-dataset `StandardScaler` normalization is used for the four corpora. The scaler files should be placed in:

```text
models/scalers/
```

## Model

The selected deployment model is **Model C_v2**, a lightweight multi-branch dilated 1D CNN.

| Item | Value |
|---|---:|
| Model | Model C_v2 |
| Architecture | Multi-branch dilated 1D CNN |
| Dilation rates | 1, 2, 3 |
| Padding | Same |
| Parameters | 130,756 |
| Validation accuracy | 95.19% |
| Test accuracy | 92.12% |
| TFLite INT8 size | 151.2 KB |
| INT8 test accuracy | 91.10% |

Model C_v3 obtains the highest combined test accuracy of **92.47%**, but Model C_v2 is selected for deployment because it provides a better trade-off among accuracy, parameter efficiency, convergence speed, and edge deployment feasibility.

## Edge and System-Level Results

| Evaluation | Result |
|---|---:|
| Raspberry Pi comprehensive edge test | 39/40 correct = 97.5% |
| Mean edge inference latency | 2.02 ms |
| Streaming duration | 125.4 minutes |
| Total streaming frames processed | 4,774 |
| Streaming mean inference latency | 2.206 ms |
| Average CPU utilization | 17.36% |
| Average RAM utilization | 17.53% |
| MQTT MDR test | 100/100 = 100.0% |
| Comprehensive lamp command latency | 213.75 ms |
| Streaming lamp actuation latency | 87.02 ms |

Note: `mqtt_publish_ms`, when present in runtime logs, records local publish-call duration from the client code and should not be interpreted as end-to-end MQTT delivery latency.

## Installation

Create and activate a Python environment:

```bash
python3 -m venv ta_env
source ta_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Raspberry Pi, `sounddevice` may require PortAudio:

```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio mosquitto mosquitto-clients
```

Start Mosquitto locally:

```bash
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

## Environment Variables

For local Raspberry Pi execution, configure these variables as needed:

```bash
export SER_MODEL_PATH=/path/to/Smart-Lighting-Speech-Emotion-Recognition/models/model_C_v2_int8.tflite
export SCALER_DIR=/path/to/Smart-Lighting-Speech-Emotion-Recognition/models/scalers
export RAVDESS_DIR=/path/to/RAVDESS
export SONOFF_IP=192.168.0.100
export SONOFF_PORT=8081
export SONOFF_DEVICE_ID=your_device_id
export SER_STREAM_LOG=/path/to/Smart-Lighting-Speech-Emotion-Recognition/logs/stream_log.jsonl
```

Do **not** commit real device IDs, Wi-Fi credentials, API keys, or `.env` files.

## Running the Scripts

### 1. Comprehensive edge test

```bash
cd src
python3 comprehensive_test.py
```

This evaluates the INT8 model on 40 RAVDESS utterances: 5 actors × 2 files × 4 emotions.

### 2. MQTT message delivery ratio test

```bash
cd src
python3 message_delivery_ratio.py
```

Expected output:

```text
MDR: 100/100 = 100.0%
```

### 3. Real-time streaming pipeline

```bash
cd src
python3 main_stream.py
```

The streaming pipeline captures microphone audio, extracts MFCC features, runs INT8 inference, stabilizes predictions using majority voting, publishes emotion events through MQTT, and sends local HTTP commands to the Sonoff B05 smart bulb.

## Smart Lighting Mapping

| Emotion | Lighting feedback | Sonoff parameters |
|---|---|---|
| Angry | Cool blue, low brightness | `color: r=0, g=100, b=255, br=40` |
| Happy | Neutral white, high brightness | `white: br=80, ct=100` |
| Neutral | Warm white, medium brightness | `white: br=70, ct=50` |
| Sad | Warm yellow, high brightness | `color: r=255, g=200, b=0, br=90` |

Gradual transitions are applied when the source and target lighting states use the same Sonoff light mode. Cross-mode transitions between color and white states are applied directly due to local API mode constraints.

## Reproducibility Notes

- Raw datasets are not included.
- Feature `.npy` files are not included because they are derived from third-party speech datasets.
- The trained INT8 TFLite model and scaler files are included to support edge inference reproduction.
- Logs are included to support reported edge, streaming, and MQTT MDR results.
- The Sonoff local HTTP API requires a device reachable on the same LAN subnet.

## Citation / Data Availability Statement

If this repository is used to support the paper's data and computer program availability section, use:

```text
The speech emotion datasets used in this study are publicly available from their original repositories: RAVDESS (Zenodo), EMODB (TU Berlin), TESS (University of Toronto), and SAVEE (University of Surrey). The dataset preparation, feature extraction, model training, ablation evaluation, TensorFlow Lite conversion notebooks, trained TFLite INT8 model, Raspberry Pi inference pipeline, MQTT integration scripts, comprehensive edge-test output, streaming runtime log, and MQTT message delivery ratio output are available at: https://github.com/bijakprasodjo/Smart-Lighting-Speech-Emotion-Recognition. The raw third-party speech datasets are not redistributed due to their original licensing terms.
```

## Author

Bijak Prasodjo  
School of Computing, Telkom University
