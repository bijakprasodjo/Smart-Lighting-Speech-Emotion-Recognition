import paho.mqtt.client as mqtt
import json
import time

BROKER_HOST = 'localhost'   
BROKER_PORT = 1883
TOPIC_EMOTION  = 'ser/emotion'   
TOPIC_LIGHTING = 'ser/lighting'  
TOPIC_STATUS   = 'ser/status'  

class MQTTPublisher:
    def __init__(self, broker=BROKER_HOST,
                 port=BROKER_PORT, client_id='SER_Edge'):
        self.broker    = broker
        self.port      = port
        self.client_id = client_id
        self.connected = False

        # Setup client
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id
        )
        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags,
                    reason_code, properties):
        if reason_code == 0:
            self.connected = True
            print(f" MQTT connected → {self.broker}:{self.port}")
        else:
            print(f" MQTT connect failed: {reason_code}")

    def _on_disconnect(self, client, userdata,
                       flags, reason_code, properties):
        self.connected = False
        print(f" MQTT disconnected: {reason_code}")

    def connect(self):
        """Connect ke broker."""
        try:
            self.client.connect(self.broker, self.port,
                                keepalive=60)
            self.client.loop_start()
            time.sleep(1)  # tunggu koneksi established
            return self.connected
        except Exception as e:
            print(f" MQTT connect error: {e}")
            return False

    def publish_result(self, inference_result,
                       timestamp=None, qos=1):
        """
        Publish hasil inferensi ke MQTT broker.
        Publish ke 2 topic: ser/emotion dan ser/lighting
        """
        if not self.connected:
            print(" MQTT tidak terkoneksi!")
            return False, 0

        ts = timestamp or time.time()

        # Payload emotion topic
        emotion_payload = {
            'timestamp'  : ts,
            'emotion'    : inference_result['emotion'],
            'confidence' : round(inference_result['confidence'], 4),
            'probabilities': inference_result['probabilities'],
            'inference_ms' : inference_result['inference_ms']
        }

        # Payload lighting topic
        lighting_payload = {
            'timestamp' : ts,
            'emotion'   : inference_result['emotion'],
            'lighting'  : inference_result['lighting']
        }

        # Publish + ukur latency
        pub_start = time.perf_counter()

        result_e = self.client.publish(
            TOPIC_EMOTION,
            json.dumps(emotion_payload),
            qos=qos
        )
        result_l = self.client.publish(
            TOPIC_LIGHTING,
            json.dumps(lighting_payload),
            qos=qos
        )

        pub_time = (time.perf_counter() - pub_start) * 1000

        success = (result_e.rc == 0 and result_l.rc == 0)
        return success, round(pub_time, 3)

    def disconnect(self):
        """Disconnect dari broker."""
        self.client.loop_stop()
        self.client.disconnect()
        print(" MQTT disconnected")