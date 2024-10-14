# exotic_matter_storage.py

from exotic_matter_generator import ExoticMatter

class ExoticMatterStorage:
    def __init__(self):
        self.exotic_matter_inventory = {}

    def store_exotic_matter(self, exotic_matter):
        """
        Store exotic matter in the inventory.
        """
        print(f"Storing {exotic_matter.matter_type} exotic matter with mass {exotic_matter.mass}...")
        self.exotic_matter_inventory[exotic_matter.matter_type] = self.exotic_matter_inventory.get(exotic_matter.matter_type, 0) + exotic_matter.mass

    def retrieve_exotic_matter(self, matter_type, mass):
        """
        Retrieve exotic matter from the inventory.
        """
        print(f"Retrieving {mass} units of {matter_type} exotic matter...")
        if matter_type in self.exotic_matter_inventory and self.exotic_matter_inventory[matter_type] >= mass:
            self.exotic_matter_inventory[matter_type] -= mass
            return ExoticMatter(matter_type, mass)
        else:
            print("Insufficient exotic matter in inventory.")
            return None

    def get_exotic_matter_inventory(self):
        """
        Get the current exotic matter inventory.
        """
        print("Exotic matter inventory:")
        for matter_type, mass in self.exotic_matter_inventory.items():
            print(f"{matter_type}: {mass} units")

### Explanation:
- **Exotic Matter Storage**: The `ExoticMatterStorage` class manages the storage and retrieval of exotic matter.

### Usage:
This module can be integrated into the overall Galactic Banking Network's exotic matter management system. The `ExoticMatterStorage` class can be used to store and retrieve exotic matter, and manage the inventory.

If you need further modifications or additional features, feel free to ask!
