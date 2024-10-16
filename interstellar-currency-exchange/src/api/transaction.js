// Import required modules
const mongoose = require('mongoose');
const logger = require('../utils/logger');
const errorHandler = require('../utils/errorHandler');
const notificationService = require('../utils/notificationService');

// Define the transaction schema
const transactionSchema = new mongoose.Schema({
  fromCurrency: { type: String, required: true },
  toCurrency: { type: String, required: true },
  amount: { type: Number, required: true },
  convertedAmount: { type: Number, required: true },
  transactionDate: { type: Date, default: Date.now },
  transactionStatus: { type: String, required: true, enum: ['pending', 'success', 'failed'] },
  transactionType: { type: String, required: true, enum: ['buy', 'sell'] }
});

// Create the transaction model
const Transaction = mongoose.model('Transaction', transactionSchema);

// Define the transaction processor
const transactionProcessor = {
  processTransaction: async (fromCurrency, toCurrency, amount, convertedAmount) => {
    try {
      // Validate the transaction details
      if (!fromCurrency || !toCurrency || !amount || !convertedAmount) {
        throw new Error('Invalid transaction details');
      }

      // Create a new transaction document
      const transaction = new Transaction({
        fromCurrency,
        toCurrency,
        amount,
        convertedAmount,
        transactionStatus: 'pending',
        transactionType: 'buy'
      });

      // Save the transaction to the database
      await transaction.save();

      // Send a notification to the user
      notificationService.sendNotification(`Transaction initiated: ${amount} ${fromCurrency} -> ${convertedAmount} ${toCurrency}`);

      // Update the transaction status to success
      transaction.transactionStatus = 'success';
      await transaction.save();

      // Return the transaction document
      return transaction;
    } catch (error) {
      // Update the transaction status to failed
      const transaction = await Transaction.findOne({ fromCurrency, toCurrency, amount, convertedAmount });
      transaction.transactionStatus = 'failed';
      await transaction.save();

      // Send a notification to the user
      notificationService.sendNotification(`Transaction failed: ${amount} ${fromCurrency} -> ${convertedAmount} ${toCurrency}`);

      // Throw the error
      throw error;
    }
  },

  getTransactionHistory: async (userId) => {
    try {
      // Find all transactions for the user
      const transactions = await Transaction.find({ userId });

      // Return the transaction history
      return transactions;
    } catch (error) {
      // Throw the error
      throw error;
    }
  },

  getTransactionDetails: async (transactionId) => {
    try {
      // Find the transaction by ID
      const transaction = await Transaction.findById(transactionId);

      // Return the transaction details
      return transaction;
    } catch (error) {
      // Throw the error
      throw error;
    }
  }
};

// Export the transaction processor
module.exports = transactionProcessor;
