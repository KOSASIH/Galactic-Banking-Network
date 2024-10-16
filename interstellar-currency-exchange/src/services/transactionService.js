// Import required modules
const mongoose = require('mongoose');
const { Transaction } = require('./transactionModel');
const { Wallet } = require('./walletModel');
const { Currency } = require('./currencyModel');
const { ExchangeRateService } = require('./exchangeRateService');

// Define the TransactionService class
class TransactionService {
  // Method to create a new transaction
  async createTransaction(fromWalletId, toWalletId, amount, currencyCode) {
    try {
      const fromWallet = await Wallet.findById(fromWalletId);
      const toWallet = await Wallet.findById(toWalletId);
      const currency = await Currency.findOne({ code: currencyCode });
      const exchangeRateService = new ExchangeRateService();

      if (!fromWallet || !toWallet || !currency) {
        throw new Error('Invalid wallet or currency');
      }

      const exchangeRate = await exchangeRateService.getExchangeRate(currencyCode, 'USD');
      const amountInUSD = amount * exchangeRate;

      const transaction = new Transaction({
        fromWalletId,
        toWalletId,
        amount,
        currencyCode,
        exchangeRate,
        amountInUSD,
        status: 'pending',
      });

      await transaction.save();
      return transaction;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  // Method to update the status of a transaction
  async updateTransactionStatus(transactionId, status) {
    try {
      const transaction = await Transaction.findById(transactionId);
      if (!transaction) {
        throw new Error('Transaction not found');
      }
      transaction.status = status;
      await transaction.save();
      return transaction;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  // Method to get all transactions for a wallet
  async getTransactionsForWallet(walletId) {
    try {
      const transactions = await Transaction.find({ $or: [{ fromWalletId: walletId }, { toWalletId: walletId }] });
      return transactions;
    } catch (error) {
      console.error(error);
      return [];
    }
  }

  // Method to get a transaction by ID
  async getTransactionById(transactionId) {
    try {
      const transaction = await Transaction.findById(transactionId);
      return transaction;
    } catch (error) {
      console.error(error);
      return null;
    }
  }
}

// Export the TransactionService class
module.exports = TransactionService;
