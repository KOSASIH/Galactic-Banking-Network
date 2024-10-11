import os
import json
from pymongo import MongoClient
from bson import ObjectId

class GalacticDataStorage:
    def __init__(self, db_name, collection_name):
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_data(self, data):
        self.collection.insert_one(data)

    def get_data(self, query):
        return self.collection.find(query)

    def update_data(self, query, update):
        self.collection.update_one(query, update)

    def delete_data(self, query):
        self.collection.delete_one(query)

    def close_connection(self):
        self.client.close()

class GalacticFileStorage:
    def __init__(self, file_path):
        self.file_path = file_path

    def write_data(self, data):
        with open(self.file_path, "w") as f:
            json.dump(data, f)

    def read_data(self):
        with open(self.file_path, "r") as f:
            return json.load(f)

def main():
    # MongoDB storage
    db_name = "galactic_db"
    collection_name = "galactic_collection"
    gds = GalacticDataStorage(db_name, collection_name)

    data = {"galaxy_type": "Spiral", "distance": 100, "velocity": 200}
    gds.insert_data(data)

    query = {"galaxy_type": "Spiral"}
    results = gds.get_data(query)
    for result in results:
        print(result)

    update = {"$set": {"distance": 150}}
    gds.update_data(query, update)

    gds.delete_data(query)

    gds.close_connection()

    # File storage
    file_path = "galactic_data.json"
    gfs = GalacticFileStorage(file_path)

    data = {"galaxy_type": "Elliptical", "distance": 50, "velocity": 100}
    gfs.write_data(data)

    data = gfs.read_data()
    print(data)

if __name__ == "__main__":
    main()
