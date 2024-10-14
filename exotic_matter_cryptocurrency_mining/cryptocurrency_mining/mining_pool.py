# mining_pool.py

from cryptocurrency_miner import CryptocurrencyMiner
from exotic_matter_generator import ExoticMatterGenerator

class MiningPool:
    def __init__(self, cryptocurrency_miner, exotic_matter_generator):
        self.cryptocurrency_miner = cryptocurrency_miner
        self.exotic_matter_generator = exotic_matter_generator
        self.miners = []

    def add_miner(self, miner):
        self.miners.append(miner)

    def distribute_mining_tasks(self, block_data):
        """
        Distribute mining tasks to multiple miners in the mining pool.
        """
        print("Distributing mining tasks...")
        for miner in self.miners:
            self.cryptocurrency_miner.mine_cryptocurrency(miner, block_data)

    def generate_exotic_matter(self, matter_type):
        """
        Generate exotic matter for all miners in the mining pool.
        """
        print("Generating exotic matter...")
        exotic_matter = self.exotic_matter_generator.generate_exotic_matter(matter_type)
        
        # Distribute exotic matter to all miners
        for miner in self.miners:
            miner.receive_exotic_matter(exotic_matter)

class Miner:
    def __init__(self, name):
        self.name = name
        self.exotic_matter = None

    def mine_cryptocurrency(self, cryptocurrency_miner, block_data):
        """
        Mine cryptocurrency using the cryptocurrency miner.
        """
        print(f"{self.name} is mining cryptocurrency...")
        cryptocurrency_miner.mine_cryptocurrency(block_data)

    def receive_exotic_matter(self, exotic_matter):
        """
        Receive exotic matter from the mining pool.
        """
        print(f"{self.name} received exotic matter: {exotic_matter.matter_type} with mass {exotic_matter.mass}")
        self.exotic_matter = exotic_matter

### Explanation:
- **Mining Pool**: The `MiningPool` class manages a group of miners, distributes mining tasks, and generates exotic matter for all miners.
- **Miner**: The `Miner` class represents a single miner in the mining pool.

### Usage:
This module can be integrated into the overall Galactic Banking Network's exotic matter cryptocurrency mining system. The `MiningPool` class can be used to manage a group of miners, distribute mining tasks, and generate exotic matter for all miners. The `Miner` class represents a single miner in the mining pool.

If you need further modifications or additional features, feel free to ask!
