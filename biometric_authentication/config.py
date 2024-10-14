# config.py

# Configuration settings for the biometric authentication system

# Neuro interface settings
NEURO_INTERFACE_DEVICE = "/dev/neuro_interface"
NEURO_INTERFACE_BAUDRATE = 9600

# Biometric data storage settings
BIOMETRIC_DATA_STORAGE = "database"  # or "file"
BIOMETRIC_DATA_ENCRYPTION_KEY = "secret_key"

# Authentication settings
AUTHENTICATION_THRESHOLD = 0.8
AUTHENTICATION_ATTEMPTS = 3

# Other settings
DEBUG_MODE = True
