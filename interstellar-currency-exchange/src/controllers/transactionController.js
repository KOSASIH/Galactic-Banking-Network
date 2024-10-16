// Import required modules
const mongoose = require('mongoose');
const Transaction = require('./dbConfig').Transaction;
const CurrencyExchange = require('./dbConfig').CurrencyExchange;
const { getExchangeRate } = require('./exchangeRateConfig');

// Transaction Controller
class TransactionController {
  // Method to create a new transaction
  static async createTransaction(req, res) {
    const { fromCurrency, toCurrency, amount } = req.body;

    try {
      const exchangeRate = await getExchangeRate(fromCurrency, toCurrency);
      const convertedAmount = amount * exchangeRate;

      const transaction = new Transaction({
        fromCurrency,
        toCurrency,
        amount,
        convertedAmount,
        transactionDate: new Date(),
        transactionStatus: 'pending',
      });

      await transaction.save();
      return res.status(201).json({ message: 'Transaction created successfully', transaction });
    } catch (error) {
      console.error('Error creating transaction:', error);
      return res.status(500).json({ message: 'Failed to create transaction' });
    }
  }

  // Method to get all transactions
  static async getTransactions(req, res) {
    try {
      const transactions = await Transaction.find({});
      return res.status(200).json(transactions);
    } catch (error) {
      console.error('Error retrieving transactions:', error);
      return res.status(500).json({ message: 'Failed to retrieve transactions' });
    }
  }

  // Method to get a specific transaction
  static async getTransaction(req, res) {
    const { id } = req.params;

    try {
      const transaction = await Transaction.findById(id);
      if (!transaction) {
        return res.status(404).json({ message: 'Transaction not found' });
      }
      return res.status(200).json(transaction);
    } catch (error) {
      console.error('Error retrieving transaction:', error);
      return res.status(500).json({ message: 'Failed to retrieve transaction' });
    }
  }

  // Method to update a transaction status
  static async updateTransactionStatus(req, res) {
    const { id } = req.params;
    const { transactionStatus } = req.body;

    try {
      const transaction = await Transaction.findById(id);
      if (!transaction) {
        return res.status(404).json({ message: 'Transaction not found' });
      }
      transaction.transactionStatus = transactionStatus;
      await transaction.save();
      return res.status(200).json({ message: 'Transaction status updated successfully' });
    } catch (error) {
      console.error('Error updating transaction status:', error);
      return res.status(500).json({ message: 'Failed to update transaction status' });
    }
  }

  // Method to delete a transaction
  static async deleteTransaction(req, res) {
    const { id } = req.params;

    try {
      const result = await Transaction.deleteOne({ _id: id });
      if (result.deletedCount === 0) {
        return res.status(404).json({ message: 'Transaction not found' });
      }
      return res.status(200).json({ message: 'Transaction deleted successfully' });
    } catch (error) {
      console.error('Error deleting transaction:', error);
      return res.status(500).json({ message: 'Failed to delete transaction' });
    }
  }
}

// Export the TransactionController class
module.exports = TransactionController;
