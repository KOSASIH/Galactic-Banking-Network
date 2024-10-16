// Import required modules
const mongoose = require('mongoose');
const { Transaction } = require('./transactionModel');
const { Wallet } = require('./walletModel');
const { Currency } = require('./currencyModel');
const { TransactionService } = require('./transactionService');

// Define the test suite for TransactionService
describe('TransactionService', () => {
  let transactionService;

  // Before each test, connect to the database and create a new transaction service
  beforeEach(async () => {
    await mongoose.connect('mongodb://localhost/test', { useNewUrlParser: true, useUnifiedTopology: true });
    transactionService = new TransactionService();

    // Create test wallets and currency
    await Wallet.create([{ _id: 'wallet_id_1', balance: 1000 }, { _id: 'wallet_id_2', balance: 500 }]);
    await Currency.create([{ code: 'USD' }, { code: 'EUR' }]);
  });

  // After each test, disconnect from the database and clean up
  afterEach(async () => {
    await Wallet.deleteMany({});
    await Currency.deleteMany({});
    await Transaction.deleteMany({});
    await mongoose.disconnect();
  });

  // Test 1: Create a new transaction
  it('should create a new transaction', async () => {
    const transaction = await transactionService.createTransaction('wallet_id_1', 'wallet_id_2', 100, 'USD');

    expect(transaction).toBeDefined();
    expect(transaction.fromWalletId).toBe('wallet_id_1');
    expect(transaction.toWalletId).toBe('wallet_id_2');
    expect(transaction.amount).toBe(100);
    expect(transaction.currencyCode).toBe('USD');
    expect(transaction.status).toBe('pending');
  });

  // Test 2: Update the status of a transaction
  it('should update the status of a transaction', async () => {
    const transaction = await transactionService.createTransaction('wallet_id_1', 'wallet_id_2', 100, 'USD');
    const updatedTransaction = await transactionService.updateTransactionStatus(transaction.id, 'completed');

    expect(updatedTransaction.status).toBe('completed');
  });

  // Test 3: Get all transactions for a wallet
  it('should get all transactions for a wallet', async () => {
    await transactionService.createTransaction('wallet_id_1', 'wallet_id_2', 100, 'USD');
    await transactionService.createTransaction('wallet_id_2', 'wallet_id_1', 50, 'USD');

    const transactions = await transactionService.getTransactionsForWallet('wallet_id_1');

    expect(transactions.length).toBe(2);
  });

  // Test 4: Get a transaction by ID
  it('should get a transaction by ID', async () => {
    const transaction = await transactionService.createTransaction('wallet_id_1', 'wallet_id_2', 100, 'USD');
    const retrievedTransaction = await transactionService.getTransactionById(transaction.id);

    expect(retrievedTransaction).toBeDefined();
    expect(retrievedTransaction.id).toBe(transaction.id);
  });

  // Test 5: Handle invalid wallet IDs during transaction creation
  it('should throw an error for invalid wallet IDs', async () => {
    await expect(transactionService.createTransaction('invalid_wallet_id', 'wallet_id_2', 100, 'USD')).rejects.toThrow('Invalid wallet or currency');
  });

  // Test 6: Handle invalid currency during transaction creation
  it('should throw an error for invalid currency', async () => {
    await expect(transactionService.createTransaction('wallet_id_1', 'wallet_id_2', 100, 'INVALID')).rejects.toThrow('Invalid wallet or currency');
  });
});
