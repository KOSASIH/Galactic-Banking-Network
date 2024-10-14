# energy_harvester.py

import random

class EnergyHarvester:
    def __init__(self, efficiency=0.85):
        """
        Initialize the energy harvester with a specified efficiency.
        """
        self.efficiency = efficiency
        self.energy_collected = 0

    def harvest_energy(self, energy_source):
        """
        Harvest energy from a specified energy source.
        """
        print(f"Harvesting energy from {energy_source}...")
        harvested_energy = self._simulate_energy_harvesting(energy_source)
        self.energy_collected += harvested_energy * self.efficiency
        print(f"Harvested {harvested_energy:.2f} units of energy (after efficiency: {self.energy_collected:.2f} units).")

    def _simulate_energy_harvesting(self, energy_source):
        """
        Simulate the energy harvesting process based on the type of energy source.
        """
        if energy_source == "solar":
            return random.uniform(50, 100)  # Simulated solar energy harvested
        elif energy_source == "wind":
            return random.uniform(30, 80)   # Simulated wind energy harvested
        elif energy_source == "geothermal":
            return random.uniform(20, 60)   # Simulated geothermal energy harvested
        else:
            print("Unknown energy source. No energy harvested.")
            return 0

    def get_collected_energy(self):
        """
        Get the total amount of energy collected.
        """
        return self.energy_collected

    def reset_energy_collected(self):
        """
        Reset the collected energy to zero.
        """
        self.energy_collected = 0
        print("Energy collected has been reset to zero.")

### Explanation:
- **Energy Harvester**: The `EnergyHarvester` class is responsible for harvesting energy from various sources (solar, wind, geothermal) and managing the efficiency of the harvesting process.
- **Energy Collection**: The class tracks the total energy collected and allows for resetting the collected energy.

### Usage:
This module can be integrated into the overall Galactic Banking Network's energy management system. The `EnergyHarvester` class can be used to harvest energy from different sources, track the amount of energy collected, and manage efficiency.

If you need further modifications or additional features, feel free to ask!
