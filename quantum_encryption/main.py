# main.py

from config import Config
from data_storage.encrypted_data_storage import EncryptedDataStorage
from data_storage.secure_data_retrieval import SecureDataRetrieval
from data_storage.data_integrity_verification import DataIntegrityVerification
from utils.quantum_utils import generate_random_bits, generate_random_basis
from utils.cryptography_utils import hash_data, generate_mac, verify_mac

def main():
    config = Config()

    # Generate random bits and basis
    alice_bits = generate_random_bits(config.key_length)
    bob_basis = generate_random_basis(config.key_length)

    # Create an encrypted data storage system
    encrypted_data_storage = EncryptedDataStorage(config.error_rate)

    # Store data in the encrypted data storage system
    data = generate_random_bits(config.data_length)
    key = generate_random_bits(config.key_length)
    encrypted_data_storage.store_data(data, key)

    # Retrieve data from the encrypted data storage system
    secure_data_retrieval = SecureDataRetrieval(config.error_rate)
    retrieved_data = secure_data_retrieval.retrieve_data(encrypted_data_storage.retrieve_data(key), key)

    # Verify the integrity of the retrieved data
    data_integrity_verification = DataIntegrityVerification(config.error_rate)
    verified_data = data_integrity_verification.verify_data_integrity(retrieved_data)

    # Hash the verified data
    hashed_data = hash_data(verified_data)

    # Generate a message authentication code (MAC) for the hashed data
    mac = generate_mac(hashed_data, key)

    # Verify the MAC
    verified_mac = verify_mac(hashed_data, key, mac)

    if verified_mac:
        print("Data integrity verified successfully!")
    else:
        print("Data integrity verification failed!")

if __name__ == "__main__":
    main()
