// database/models/Transaction.js

const mongoose = require('mongoose');

// Define the transaction schema
const transactionSchema = new mongoose.Schema({
    transactionId: {
        type: Number,
        required: true,
        unique: true,
        index: true
    },
    sender: {
        type: String,
        required: true,
        trim: true
    },
    recipient: {
        type: String,
        required: true,
        trim: true
    },
    amount: {
        type: Number,
        required: true,
        min: 0
    },
    currency: {
        type: String,
        required: true,
        trim: true
    },
    planet: {
        type: String,
        required: true,
        trim: true
    },
    status: {
        type: String,
        enum: ['pending', 'completed', 'failed'],
        default: 'pending'
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

// Create a model from the schema
const Transaction = mongoose.model('Transaction', transactionSchema);

// Export the model
module.exports = Transaction;
