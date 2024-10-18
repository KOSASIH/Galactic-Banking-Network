// src/transactions/transactionService.js
const Transaction = require('./transactionModel');
const CurrencyService = require('./currencyService'); // Assuming you have a CurrencyService for exchange rates
const logger = require('../utils/logger');

class TransactionService {
    // Create a new transaction
    async createTransaction(userId, fromCurrency, toCurrency, amount) {
        try {
            // Fetch the exchange rate
            const exchangeRate = await CurrencyService.getExchangeRate(fromCurrency, toCurrency);
            if (!exchangeRate.success) {
                throw new Error(exchangeRate.message);
            }

            // Create a new transaction
            const transaction = new Transaction({
                userId,
                fromCurrency,
                toCurrency,
                amount,
                exchangeRate: exchangeRate.rate
            });

            // Calculate transaction fee (e.g., 2%)
            transaction.calculateTransactionFee(2); // You can make this dynamic based on business logic

            // Save the transaction to the database
            await transaction.save();
            logger.info(`Transaction created successfully: ${transaction}`);
            return transaction;
        } catch (error) {
            logger.error(`Failed to create transaction: ${error.message}`);
            throw new Error(`Transaction creation failed: ${error.message}`);
        }
    }

    // Retrieve transaction history for a user
    async getTransactionHistory(userId) {
        try {
            const transactions = await Transaction.find({ userId }).sort({ createdAt: -1 });
            logger.info(`Retrieved transaction history for user ${userId}`);
            return transactions;
        } catch (error) {
            logger.error(`Failed to retrieve transaction history: ${error.message}`);
            throw new Error(`Could not retrieve transaction history: ${error.message}`);
        }
    }

    // Get a specific transaction by ID
    async getTransactionById(transactionId) {
        try {
            const transaction = await Transaction.findById(transactionId);
            if (!transaction) {
                throw new Error('Transaction not found');
            }
            logger.info(`Retrieved transaction: ${transaction}`);
            return transaction;
        } catch (error) {
            logger.error(`Failed to retrieve transaction: ${error.message}`);
            throw new Error(`Could not retrieve transaction: ${error.message}`);
        }
    }
}

module.exports = new TransactionService();
