# gw_data_encoding.py

import base64

def encode_signal(signal):
    # Encode gravitational wave signal using Base64 encoding
    encoded_signal = base64.b64encode(signal)
    return encoded_signal

def decode_signal(encoded_signal):
    # Decode gravitational wave signal using Base64 decoding
    decoded_signal = base64.b64decode(encoded_signal)
    return decoded_signal
