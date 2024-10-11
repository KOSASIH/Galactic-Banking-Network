import socket
import threading
import time
import json

class GalacticNetworkMonitor:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.clients = []
        self.lock = threading.Lock()

    def handle_client(self, client_socket):
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break
                self.process_data(data)
            except Exception as e:
                print(f"Error handling client: {e}")
                break
        client_socket.close()

    def process_data(self, data):
        data = json.loads(data.decode('utf-8'))
        print(f"Received data from {data['source']}: {data['data']}")

    def broadcast(self, data):
        with self.lock:
            for client in self.clients:
                client.sendall(json.dumps(data).encode('utf-8'))

    def start(self):
        print(f"Galactic Network Monitor started on {self.host}:{self.port}")
        while True:
            client_socket, address = self.socket.accept()
            self.clients.append(client_socket)
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def stop(self):
        self.socket.close()
        print("Galactic Network Monitor stopped")

def main():
    monitor = GalacticNetworkMonitor('localhost', 8080)
    try:
        monitor.start()
    except KeyboardInterrupt:
        monitor.stop()

if __name__ == "__main__":
    main()
