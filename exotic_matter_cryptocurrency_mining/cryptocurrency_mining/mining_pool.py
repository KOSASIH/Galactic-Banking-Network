# mining_pool.py

class MiningPool:
    def __init__(self, miners):
        self.miners = miners

    def add_miner(self, miner):
        """
        Add a miner to the mining pool.

        Parameters:
        - miner: Miner to be added.
        """
        self.miners.append(miner)

    def remove_miner(self, miner):
        """
        Remove a miner from the mining pool.

        Parameters:
        - miner: Miner to be removed.
        """
        self.miners.remove(miner)

    def mine_block(self, block_data):
        """
        Mine a block using the mining pool.

        Parameters:
        - block_data: Block data to be mined.

        Returns:
        - mined_block: Mined block.
        """
        for miner in self.miners:
            mined_block = miner.mine_block(block_data)
            if mined_block:
                return mined_block
        return None
