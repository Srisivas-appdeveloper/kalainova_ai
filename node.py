import hivemind
import time

print("Starting Hivemind node...")
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/59640"],
    start=True
)

print("✅ Node started!")
print("Your Node ID:", dht.peer_id)
print("Share this address with other devices to connect:")
for addr in dht.get_visible_maddrs():
    print(" →", addr)

print("\nNode is running... Press Ctrl+C to stop")
while True:
    time.sleep(5)
    print(f"Connected peers: {len(dht.get_visible_maddrs())}")
