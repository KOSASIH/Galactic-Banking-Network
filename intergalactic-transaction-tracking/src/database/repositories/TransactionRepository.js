// database/repositories/TransactionRepository.js

const Transaction = require('../models/Transaction');

class TransactionRepository {
    // Create a new transaction
    async createTransaction(transactionData) {
        try {
            const transaction = new Transaction(transactionData);
            await transaction.save();
            return transaction;
        } catch (error) {
            throw new Error(`Error creating transaction: ${error.message}`);
        }
    }

    // Get a transaction by ID
    async getTransactionById(transactionId) {
        try {
            const transaction = await Transaction.findOne({ transactionId });
            if (!transaction) {
                throw new Error('Transaction not found');
            }
            return transaction;
        } catch (error) {
            throw new Error(`Error retrieving transaction: ${error.message}`);
        }
    }

    // Get all transactions with optional filtering and pagination
    async getAllTransactions(filter = {}, options = {}) {
        const { page = 1, limit = 10 } = options;
        try {
            const transactions = await Transaction.find(filter)
                .skip((page - 1) * limit)
                .limit(limit)
                .sort({ createdAt: -1 });
            const total = await Transaction.countDocuments(filter);
            return { transactions, total, page, totalPages: Math.ceil(total / limit) };
        } catch (error) {
            throw new Error(`Error retrieving transactions: ${error.message}`);
        }
    }

    // Update a transaction by ID
    async updateTransaction(transactionId, updateData) {
        try {
            const transaction = await Transaction.findOneAndUpdate(
                { transactionId },
                updateData,
                { new: true, runValidators: true }
            );
            if (!transaction) {
                throw new Error('Transaction not found');
            }
            return transaction;
        } catch (error) {
            throw new Error(`Error updating transaction: ${error.message}`);
        }
    }

    // Delete a transaction by ID
    async deleteTransaction(transactionId) {
        try {
            const transaction = await Transaction.findOneAndDelete({ transactionId });
            if (!transaction) {
                throw new Error('Transaction not found');
            }
            return transaction;
        } catch (error) {
            throw new Error(`Error deleting transaction: ${error.message}`);
        }
    }
}

module.exports = new TransactionRepository();
