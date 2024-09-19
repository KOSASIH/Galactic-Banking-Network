import os
import sys
from gbn.database import Database
from gbn.network import Node
from gbn.security import Authentication

def main():
    config = load_config()
    database = Database(config)
    node = Node(config)
    authentication = Authentication(config)

    # Start the node and database
    node.start()
    database.start()

    # Authenticate and authorize users
    while True:
        username = input("Enter username: ")
        password = input("Enter password: ")
        user_id = authentication.authenticate(username, password)
        if user_id:
            print("Authenticated!")
            # Authorize user and grant access to the network
            pass
        else:
            print("Invalid credentials")

if __name__ == "__main__":
    main()
