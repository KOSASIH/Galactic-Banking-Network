import datetime

class Transaction:
    def __init__(self, account_id, transaction_type, amount):
        self.account_id = account_id
        self.transaction_type = transaction_type
        self.amount = amount
        self.timestamp = datetime.datetime.now()
