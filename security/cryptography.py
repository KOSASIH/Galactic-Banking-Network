import cryptography.fernet

def generate_encryption_key():
    return cryptography.fernet.Fernet.generate_key()
