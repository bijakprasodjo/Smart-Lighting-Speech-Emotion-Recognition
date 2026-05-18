import os
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter

LABEL_MAP = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}

MODEL_PATH = os.getenv(
    'SER_MODEL_PATH',
    '/home/bijakprasodjo/TA_SER/models/model_C_v2_int8.tflite'
)

LIGHTING_RULES = {
    'angry'  : {'color': 'blue',   'brightness': 40,
                'cct': 6500,
                'reason': 'Cool blue untuk menenangkan'},
    'happy'  : {'color': 'white',  'brightness': 80,
                'cct': 5000,
                'reason': 'Netral untuk mempertahankan mood'},
    'neutral': {'color': 'warm',   'brightness': 70,
                'cct': 4000,
                'reason': 'Warm white untuk fokus'},
    'sad'    : {'color': 'yellow', 'brightness': 90,
                'cct': 3000,
                'reason': 'Warm bright untuk meningkatkan mood'},
}

class SERInference:
    def __init__(self, model_path=MODEL_PATH):
       
        print(f"  Loading model: {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_idx  = self.input_details[0]['index']
        self.output_idx = self.output_details[0]['index']

        self.is_int8 = (
            self.input_details[0]['dtype'] == np.int8
        )

        if self.is_int8:
            self.input_scale      = self.input_details[0]['quantization'][0]
            self.input_zero_point = self.input_details[0]['quantization'][1]
            self.output_scale      = self.output_details[0]['quantization'][0]
            self.output_zero_point = self.output_details[0]['quantization'][1]

        print(f"  Model loaded! INT8={self.is_int8}")
        print(f"  Input  : {self.input_details[0]['shape']} "
              f"dtype={self.input_details[0]['dtype']}")
        print(f"  Output : {self.output_details[0]['shape']} "
              f"dtype={self.output_details[0]['dtype']}")

    def predict(self, features):
       
        features = features.astype(np.float32)

        expected_shape = tuple(self.input_details[0]['shape'])
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"Invalid input shape: got {features.shape}, expected {expected_shape}"
            )

        if self.is_int8:
            quantized = features / self.input_scale + self.input_zero_point
            quantized = np.clip(quantized, -128, 127)
            input_data = quantized.astype(np.int8)
        else:
            input_data = features

        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_idx)
        inference_time = (time.perf_counter() - start) * 1000  
        
        if self.is_int8:
            probs = (
                (raw_output.astype(np.float32) - self.output_zero_point)
                * self.output_scale
            )[0]
        else:
            probs = raw_output[0]

        pred_idx   = int(np.argmax(probs))
        pred_label = LABEL_MAP[pred_idx]
        confidence = float(probs[pred_idx])
        lighting   = LIGHTING_RULES[pred_label]

        return {
            'emotion'       : pred_label,
            'confidence'    : confidence,
            'probabilities' : {
                LABEL_MAP[i]: float(probs[i])
                for i in range(len(probs))
            },
            'lighting'      : lighting,
            'inference_ms'  : round(inference_time, 3)
        }