// src/transactions/transactionController.js
const TransactionService = require('./transactionService');
const { validationResult } = require('express-validator');
const logger = require('../utils/logger');

class TransactionController {
    // Create a new transaction
    async createTransaction(req, res) {
        try {
            // Validate request input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({ errors: errors.array() });
            }

            const { userId, fromCurrency, toCurrency, amount } = req.body;

            // Create the transaction
            const transaction = await TransactionService.createTransaction(userId, fromCurrency, toCurrency, amount);
            return res.status(201).json({ success: true, transaction });
        } catch (error) {
            logger.error(`Error creating transaction: ${error.message}`);
            return res.status(500).json({ success: false, message: error.message });
        }
    }

    // Retrieve transaction history for a user
    async getTransactionHistory(req, res) {
        try {
            const { userId } = req.params;
            const transactions = await TransactionService.getTransactionHistory(userId);
            return res.status(200).json({ success: true, transactions });
        } catch (error) {
            logger.error(`Error retrieving transaction history: ${error.message}`);
            return res.status(500).json({ success: false, message: error.message });
        }
    }

    // Get a specific transaction by ID
    async getTransactionById(req, res) {
        try {
            const { transactionId } = req.params;
            const transaction = await TransactionService.getTransactionById(transactionId);
            return res.status(200).json({ success: true, transaction });
        } catch (error) {
            logger.error(`Error retrieving transaction: ${error.message}`);
            return res.status(500).json({ success: false, message: error.message });
        }
    }
}

module.exports = new TransactionController();
