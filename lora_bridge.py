import asyncio
import websockets
import serial
import sys
import json
import threading
import paho.mqtt.client as paho_mqtt

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# Change SERIAL_PORT to your Pico's actual COM port
SERIAL_PORT  = 'COM3'
BAUD_RATE    = 115200
WS_HOST      = 'localhost'
WS_PORT      = 8765

# MQTT cloud config — must match the session name in index.html / manager.html
MQTT_BROKER  = 'broker.hivemq.com'
MQTT_PORT    = 1883
MQTT_SESSION = 'bask-demo-01'
MQTT_GPS     = f'bask-solutions/{MQTT_SESSION}/gps'

# ─── WEBSOCKET BROADCAST STATE ───────────────────────────────────────────────
CONNECTED_CLIENTS = set()

async def register(websocket):
    """Register a new dashboard client (local WebSocket)."""
    CONNECTED_CLIENTS.add(websocket)
    print(f"[Bridge-WS] Client connected. Total: {len(CONNECTED_CLIENTS)}")
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        print(f"[Bridge-WS] Client disconnected. Total: {len(CONNECTED_CLIENTS)}")

async def broadcast(message):
    """Push a message to every connected local WebSocket client."""
    if not CONNECTED_CLIENTS:
        return
    await asyncio.gather(
        *[client.send(message) for client in CONNECTED_CLIENTS],
        return_exceptions=True
    )

# ─── MQTT CLIENT (runs in a background thread via paho) ──────────────────────
mqtt_client = None

def mqtt_init():
    """Initialise paho MQTT client and connect to HiveMQ cloud broker."""
    global mqtt_client
    client_id = f'bask-bridge-{__import__("random").randint(1000, 9999)}'
    mqtt_client = paho_mqtt.Client(client_id=client_id)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"[Bridge-MQTT] Connected to {MQTT_BROKER}. Publishing to {MQTT_GPS}")
        else:
            print(f"[Bridge-MQTT] Connection failed (rc={rc}). Will retry...")

    def on_disconnect(client, userdata, rc):
        print(f"[Bridge-MQTT] Disconnected (rc={rc}). Auto-reconnecting...")

    mqtt_client.on_connect    = on_connect
    mqtt_client.on_disconnect = on_disconnect

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        # loop_start() runs the MQTT network loop in a background thread
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[Bridge-MQTT] Initial connect failed: {e}. Continuing without MQTT.")

def mqtt_publish(lat: float, lng: float):
    """Publish a GPS reading to the HiveMQ cloud broker."""
    if mqtt_client is None:
        return
    try:
        payload = json.dumps({"lat": lat, "lng": lng, "source": "lora"})
        mqtt_client.publish(MQTT_GPS, payload, qos=0, retain=False)
    except Exception as e:
        print(f"[Bridge-MQTT] Publish error: {e}")

# ─── SERIAL READER ───────────────────────────────────────────────────────────
async def serial_reader():
    """
    Background task: reads the LoRa/Pico serial port and:
      1. Broadcasts raw "lat,lng" string to all local WebSocket clients
      2. Publishes structured JSON to the HiveMQ MQTT cloud broker

    Expected input format from Pico:  "11.01234,76.95678"
    """
    print(f"[Bridge] Attempting serial connection on {SERIAL_PORT} @ {BAUD_RATE} baud...")

    while True:  # Outer loop: reconnect on hardware disconnect
        ser = None
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"[Bridge] LoRa hardware ONLINE on {SERIAL_PORT}. Streaming GPS data...")

            while True:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8', errors='replace').strip()
                    except Exception:
                        line = ''

                    if line:
                        print(f"[LoRa Rx] -> {line}")

                        # 1. Broadcast raw string to all local WS clients
                        await broadcast(line)

                        # 2. Parse and publish to MQTT cloud
                        try:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                lat = float(parts[0])
                                lng = float(parts[1])
                                mqtt_publish(lat, lng)
                        except ValueError:
                            pass  # Not a valid lat,lng — skip MQTT publish

                await asyncio.sleep(0.05)  # 50 ms poll — ~20 Hz max

        except serial.SerialException as e:
            error_msg = f"ERROR:Hardware Disconnected - {e}"
            print(f"[Bridge] {error_msg}")
            await broadcast(error_msg)
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            print(f"[Bridge] Retrying {SERIAL_PORT} in 5 seconds...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[Bridge] Unexpected serial error: {e}")
            await asyncio.sleep(5)

# ─── MAIN ────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 55)
    print("  BASK Edge-Bridge v3.0 — Dual-Mode (WS + MQTT)")
    print(f"  WebSocket Server : ws://{WS_HOST}:{WS_PORT}  (local)")
    print(f"  MQTT Broker      : {MQTT_BROKER}:{MQTT_PORT}  (cloud)")
    print(f"  MQTT Topic       : {MQTT_GPS}")
    print(f"  Serial Port      : {SERIAL_PORT} @ {BAUD_RATE} baud")
    print("=" * 55)

    # Start MQTT connection in background (non-blocking)
    mqtt_init()

    # Run WebSocket server + serial reader concurrently
    async with websockets.serve(register, WS_HOST, WS_PORT):
        print(f"[Bridge] WebSocket server ACTIVE. Waiting for dashboard clients...")
        await serial_reader()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("\n[Bridge] Graceful shutdown. Goodbye.")
        sys.exit(0)