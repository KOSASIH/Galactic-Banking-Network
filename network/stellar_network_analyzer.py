import os
import hashlib
import json
from stellar_sdk import Server, Keypair, TransactionBuilder, Network
from stellar_sdk.horizon import Horizon
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class StellarNetworkAnalyzer:
  def __init__(self, network_passphrase, horizon_url):
    self.network_passphrase = network_passphrase
    self.horizon_url = horizon_url
    self.server = Server(horizon_url)
    self.horizon = Horizon(horizon_url)

  def create_keypair(self):
    keypair = Keypair.random()
    return keypair

  def create_account(self, keypair):
    account = self.server.load_account(keypair.public_key)
    if not account:
      transaction = TransactionBuilder(
        source_account=keypair.public_key,
        network_passphrase=self.network_passphrase,
        base_fee=100
      ).append_create_account_op(
        destination=keypair.public_key,
        starting_balance="1000"
      ).build()
      transaction.sign(keypair)
      response = self.server.submit_transaction(transaction)
      print(f"Account created: {response}")

  def send_asset(self, source_keypair, destination_keypair, asset_code, amount):
    transaction = TransactionBuilder(
      source_account=source_keypair.public_key,
      network_passphrase=self.network_passphrase,
      base_fee=100
    ).append_payment_op(
      destination=destination_keypair.public_key,
      asset_code=asset_code,
      amount=amount
    ).build()
    transaction.sign(source_keypair)
    response = self.server.submit_transaction(transaction)
    print(f"Asset sent: {response}")

  def get_account_balance(self, keypair):
    account = self.server.load_account(keypair.public_key)
    balance = account.balances[0].balance
    print(f"Account balance: {balance}")

  def collect_data(self):
    # Collect data from the Stellar network
    data = []
    for account in self.server.accounts().call():
      data.append({
        "account_id": account.id,
        "balance": account.balances[0].balance,
        "sequence": account.sequence,
        "last_modified_ledger": account.last_modified_ledger
      })
    return pd.DataFrame(data)

  def build_network(self, data):
    # Build a network graph using the collected data
    G = nx.Graph()
    for index, row in data.iterrows():
      G.add_node(row["account_id"])
      for other_index, other_row in data.iterrows():
        if row["account_id"] != other_row["account_id"]:
          G.add_edge(row["account_id"], other_row["account_id"])
    return G

  def analyze_network(self, G):
    # Analyze the network graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {nx.degree(G).values().mean()}")
    print(f"Clustering coefficient: {nx.clustering(G).values().mean()}")

  def visualize_network(self, G):
    # Visualize the network graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, width=1, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.show()

# Example usage:
if __name__ == '__main__':
  network_passphrase = "Test SDF Network ; September 2015"
  horizon_url = "https://horizon-testnet.stellar.org"

  stellar_network_analyzer = StellarNetworkAnalyzer(network_passphrase, horizon_url)

  source_keypair = stellar_network_analyzer.create_keypair()
  destination_keypair = stellar_network_analyzer.create_keypair()

  stellar_network_analyzer.create_account(source_keypair)
  stellar_network_analyzer.create_account(destination_keypair)

  data = stellar_network_analyzer.collect_data()
  G = stellar_network_analyzer.build_network(data)
  stellar_network_analyzer.analyze_network(G)
  stellar_network_analyzer.visualize_network(G)
