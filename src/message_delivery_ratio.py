import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "ser/test"
QOS = 1
N_MESSAGES = 100

received = []

def on_message(client, userdata, msg):
    received.append(msg.payload.decode())

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message

client.connect(BROKER_HOST, BROKER_PORT)
client.subscribe(TOPIC, qos=QOS)
client.loop_start()

time.sleep(1)

for i in range(N_MESSAGES):
    info = client.publish(TOPIC, f"msg_{i}", qos=QOS)
    info.wait_for_publish()
    time.sleep(0.1)

time.sleep(2)

delivered = len(received)
mdr = delivered / N_MESSAGES * 100

print(f"MDR: {delivered}/{N_MESSAGES} = {mdr:.1f}%")

result = {
    "timestamp": datetime.now().isoformat(),
    "broker_host": BROKER_HOST,
    "broker_port": BROKER_PORT,
    "topic": TOPIC,
    "qos": QOS,
    "attempted_messages": N_MESSAGES,
    "delivered_messages": delivered,
    "message_delivery_ratio_percent": round(mdr, 1)
}

with open("mqtt_mdr_100_messages_output.json", "w") as f:
    json.dump(result, f, indent=2)

with open("mqtt_mdr_100_messages_output.txt", "w") as f:
    f.write(f"MDR: {delivered}/{N_MESSAGES} = {mdr:.1f}%\n")

client.loop_stop()
client.disconnect()