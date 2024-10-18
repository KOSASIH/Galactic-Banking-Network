// src/transactions/transactionModel.js
const mongoose = require('mongoose');

// Define the Transaction schema
const transactionSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
        ref: 'User ' // Assuming you have a User model
    },
    fromCurrency: {
        type: String,
        required: true,
        trim: true
    },
    toCurrency: {
        type: String,
        required: true,
        trim: true
    },
    amount: {
        type: Number,
        required: true,
        min: 0
    },
    convertedAmount: {
        type: Number,
        required: true,
        min: 0
    },
    exchangeRate: {
        type: Number,
        required: true,
        min: 0
    },
    transactionFee: {
        type: Number,
        default: 0,
        min: 0
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

// Method to calculate transaction fee
transactionSchema.methods.calculateTransactionFee = function (feePercentage) {
    this.transactionFee = (this.amount * feePercentage) / 100;
    this.convertedAmount -= this.transactionFee; // Deduct fee from converted amount
};

// Create the Transaction model
const Transaction = mongoose.model('Transaction', transactionSchema);

module.exports = Transaction;
