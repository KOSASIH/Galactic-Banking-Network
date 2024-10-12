import os
import hashlib
import json
from stellar_sdk import Server, Keypair, TransactionBuilder, Network
from stellar_sdk.horizon import Horizon

class StellarBlockchainScalability:
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

  def get_ledger(self):
    ledger = self.horizon.ledgers().call()
    print(f"Ledger: {ledger}")

  def get_transaction(self, transaction_hash):
    transaction = self.horizon.transactions().transaction(transaction_hash).call()
    print(f"Transaction: {transaction}")

  def get_account_transactions(self, account_id):
    transactions = self.horizon.accounts().account_id(account_id).transactions().call()
    print(f"Transactions: {transactions}")

# Example usage:
if __name__ == '__main__':
  network_passphrase = "Test SDF Network ; September 2015"
  horizon_url = "https://horizon-testnet.stellar.org"

  stellar_blockchain_scalability = StellarBlockchainScalability(network_passphrase, horizon_url)

  source_keypair = stellar_blockchain_scalability.create_keypair()
  destination_keypair = stellar_blockchain_scalability.create_keypair()

  stellar_blockchain_scalability.create_account(source_keypair)
  stellar_blockchain_scalability.create_account(destination_keypair)

  stellar_blockchain_scalability.send_asset(source_keypair, destination_keypair, "USD", "10.0")

  stellar_blockchain_scalability.get_account_balance(destination_keypair)

  stellar_blockchain_scalability.get_ledger()

  transaction_hash = "your_transaction_hash"
  stellar_blockchain_scalability.get_transaction(transaction_hash)

  account_id = "your_account_id"
  stellar_blockchain_scalability.get_account_transactions(account_id)
