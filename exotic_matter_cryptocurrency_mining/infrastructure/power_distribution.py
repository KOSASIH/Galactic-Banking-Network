# power_distribution.py

from energy_harvester import EnergyHarvester

class PowerDistribution:
    def __init__(self, energy_harvester):
        """
        Initialize the power distribution system with an energy harvester.
        """
        self.energy_harvester = energy_harvester
        self.power_grid = {}

    def add_device_to_power_grid(self, device_name, power_requirement):
        """
        Add a device to the power grid with its power requirement.
        """
        print(f"Adding {device_name} to the power grid with a power requirement of {power_requirement} units...")
        self.power_grid[device_name] = power_requirement

    def distribute_power(self):
        """
        Distribute power to all devices in the power grid.
        """
        print("Distributing power to all devices in the power grid...")
        total_power_required = sum(self.power_grid.values())
        if self.energy_harvester.get_collected_energy() >= total_power_required:
            for device_name, power_requirement in self.power_grid.items():
                print(f"Distributing {power_requirement} units of power to {device_name}...")
        else:
            print("Insufficient energy to distribute power to all devices.")

    def get_power_grid_status(self):
        """
        Get the current status of the power grid.
        """
        print("Power grid status:")
        for device_name, power_requirement in self.power_grid.items():
            print(f"{device_name}: {power_requirement} units")

    def remove_device_from_power_grid(self, device_name):
        """
        Remove a device from the power grid.
        """
        print(f"Removing {device_name} from the power grid...")
        if device_name in self.power_grid:
            del self.power_grid[device_name]
        else:
            print(f"{device_name} not found in the power grid.")

### Explanation:
- **Power Distribution**: The `PowerDistribution` class manages the distribution of power to various devices in the power grid.
- **Power Grid**: The class tracks the power requirements of all devices in the power grid and ensures that sufficient energy is available to distribute power to all devices.

### Usage:
This module can be integrated into the overall Galactic Banking Network's power management system. The `PowerDistribution` class can be used to manage the power grid, distribute power to devices, and track the status of the power grid.

If you need further modifications or additional features, feel free to ask!

Example usage:
```python
energy_harvester = EnergyHarvester()
power_distribution = PowerDistribution(energy_harvester)

power_distribution.add_device_to_power_grid("Device 1", 100)
power_distribution.add_device_to_power_grid("Device 2", 200)

energy_harvester.harvest_energy("solar")
power_distribution.distribute_power()

power_distribution.get_power_grid_status()
