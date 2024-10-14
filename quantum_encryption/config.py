# config.py

class Config:
    def __init__(self):
        self.error_rate = 0.01
        self.shots = 1024
        self.backend = 'qasm_simulator'
        self.provider = 'Aer'
        self.key_length = 1024
        self.data_length = 1024
        self.hash_function = 'sha256'
        self.mac_function = 'hmac'
