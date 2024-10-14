# gw_node.py

import gw_detector
import gw_transceiver
import gw_utils.gw_math as gw_math

class GWNode:
    def __init__(self, node_id, detector, transceiver):
        self.node_id = node_id
        self.detector = detector
        self.transceiver = transceiver
        self.signal_buffer = []

    def receive_signal(self, signal):
        # Receive gravitational wave signal using the detector
        detected_signal = self.detector.receive_signal(signal)
        self.signal_buffer.append(detected_signal)

    def transmit_signal(self, signal):
        # Transmit gravitational wave signal using the transceiver
        transmitted_signal = self.transceiver.transmit_signal(signal)
        return transmitted_signal

    def process_signal(self, signal):
        # Process the received signal using signal processing techniques
        processed_signal = gw_math.filter_signal(signal)
        processed_signal = gw_math.amplify_signal(processed_signal)
        return processed_signal

    def get_signal_buffer(self):
        return self.signal_buffer

    def clear_signal_buffer(self):
        self.signal_buffer = []
