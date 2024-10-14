# dimension_map.py

class DimensionMap:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def get_dimension(self, dimension_id):
        """
        Get a dimension by ID.

        Parameters:
        - dimension_id: ID of the dimension to get.

        Returns:
        - dimension: Dimension.
        """
        # Implement dimension retrieval algorithm here
        return self.dimensions[dimension_id]
