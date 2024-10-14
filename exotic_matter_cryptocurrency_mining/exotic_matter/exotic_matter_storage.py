# exotic_matter_storage.py

class ExoticMatterStorage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.exotic_matter = []

    def store_exotic_matter(self, exotic_matter):
        """
        Store exotic matter.

        Parameters:
        - exotic_matter: Exotic matter to be stored.
        """
        if len(self.exotic_matter) < self.capacity:
            self.exotic_matter.append(exotic_matter)

    def retrieve_exotic_matter(self):
        """
        Retrieve exotic matter.

        Returns:
        - exotic_matter: Exotic matter.
        """
        return self.exotic_matter.pop(0)
