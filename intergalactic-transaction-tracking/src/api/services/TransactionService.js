// src/api/services/TransactionService.js

const TransactionRepository = require('../repositories/TransactionRepository');
const BlockchainService = require('../blockchain/services/BlockchainService');
const NotificationService = require('../notifications/services/NotificationService');
const { v4: uuidv4 } = require('uuid');

class TransactionService {
  // Method to get a transaction by ID
  async getTransaction(transactionId) {
    try {
      const transaction = await TransactionRepository.findById(transactionId);
      if (!transaction) {
        throw new Error('Transaction not found');
      }
      return transaction;
    } catch (error) {
      throw new Error(`Failed to retrieve transaction: ${error.message}`);
    }
  }

  // Method to create a new transaction
  async createTransaction(transactionData) {
    try {
      // Validate transaction data
      if (!transactionData.sender || !transactionData.recipient || !transactionData.amount || !transactionData.currency || !transactionData.planet) {
        throw new Error('Invalid transaction data');
      }

      // Generate a unique transaction ID
      const transactionId = uuidv4();

      // Create a new transaction
      const transaction = {
        transactionId,
        sender: transactionData.sender,
        recipient: transactionData.recipient,
        amount: transactionData.amount,
        currency: transactionData.currency,
        planet: transactionData.planet,
        status: 'pending',
      };

      // Save the transaction to the database
      await TransactionRepository.save(transaction);

      // Deploy the transaction to the blockchain
      await BlockchainService.deployTransaction(transaction);

      // Send a notification for the transaction update
      await NotificationService.sendTransactionUpdate(transaction);

      return transaction;
    } catch (error) {
      throw new Error(`Failed to create transaction: ${error.message}`);
    }
  }

  // Method to update a transaction status
  async updateTransactionStatus(transactionId, newStatus) {
    try {
      // Validate the transaction status
      if (!['pending', 'completed', 'failed'].includes(newStatus)) {
        throw new Error('Invalid transaction status');
      }

      // Retrieve the transaction from the database
      const transaction = await TransactionRepository.findById(transactionId);
      if (!transaction) {
        throw new Error('Transaction not found');
      }

      // Update the transaction status
      transaction.status = newStatus;

      // Save the updated transaction to the database
      await TransactionRepository.save(transaction);

      return transaction;
    } catch (error) {
      throw new Error(`Failed to update transaction status: ${error.message}`);
    }
  }

  // Method to get all transactions for a specific sender
  async getTransactionsBySender(senderId) {
    try {
      const transactions = await TransactionRepository.findBySender(senderId);
      return transactions;
    } catch (error) {
      throw new Error(`Failed to retrieve transactions by sender: ${error.message}`);
    }
  }
}

module.exports = new TransactionService();
