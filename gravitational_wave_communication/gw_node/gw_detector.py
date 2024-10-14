# gw_detector.py

import gw_utils.gw_math as gw_math

class GWDetector:
    def __init__(self, detector_id):
        self.detector_id = detector_id
        self.sensitivity = 0.1  # adjustable sensitivity parameter

    def receive_signal(self, signal):
        # Receive gravitational wave signal using the detector
        detected_signal = gw_math.detect_signal(signal, self.sensitivity)
        return detected_signal
