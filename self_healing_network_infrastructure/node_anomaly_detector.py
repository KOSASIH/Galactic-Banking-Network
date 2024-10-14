# gw_node_anomaly_detector.py

import network
import utils
import statistics

class GwNodeAnomalyDetector:
    def __init__(self, node):
        self.node = node

    def detect_anomaly(self):
        """
        Detect anomalies in node behavior using statistical methods.
        This method will analyze the node's historical data and current behavior to identify potential anomalies.
        """
        print("Detecting anomalies in node behavior...")
        historical_data = self.node.get_historical_data()
        current_data = self.node.get_current_data()

        # Calculate statistical metrics for historical data
        mean = statistics.mean(historical_data)
        std_dev = statistics.stdev(historical_data)

        # Check for anomalies in current data
        for data_point in current_data:
            if abs(data_point - mean) > 2 * std_dev:
                print("Anomaly detected! Node behavior is outside 2 standard deviations.")
                return True

        print("No anomalies detected. Node behavior is within normal range.")
        return False
