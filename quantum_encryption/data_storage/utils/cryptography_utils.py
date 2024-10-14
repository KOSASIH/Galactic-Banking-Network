# cryptography_utils.py

import hashlib
import hmac

def hash_data(data):
    """
    Hash data.

    Parameters:
    - data: Data to be hashed.

    Returns:
    - hashed_data: Hashed data.
    """
    return hashlib.sha256(data).hexdigest()

def generate_mac(data, key):
    """
    Generate a message authentication code (MAC).

    Parameters:
    - data: Data to be authenticated.
    - key: Key for authentication.

    Returns:
    - mac: MAC.
    """
    return hmac.new(key, data, hashlib.sha256).hexdigest()

def verify_mac(data, key, mac):
    """
    Verify a message authentication code (MAC).

    Parameters:
    - data: Data to be verified.
    - key: Key for verification.
    - mac: MAC to be verified.

    Returns:
    - verified: Whether the MAC is verified.
    """
    expected_mac = generate_mac(data, key)
    return hmac.compare_digest(mac, expected_mac)
