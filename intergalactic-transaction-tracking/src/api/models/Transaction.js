// src/api/models/Transaction.js

const mongoose = require('mongoose');

// Define the transaction schema
const transactionSchema = new mongoose.Schema({
  sender: { type: String, required: true }, // Address of the sender
  recipient: { type: String, required: true }, // Address of the recipient
  amount: { type: Number, required: true, min: 0 }, // Amount being transferred
  currency: { type: String, required: true }, // Currency type (e.g., "Galactic Credits")
  status: { 
    type: String, 
    enum: ['pending', 'completed', 'failed'], 
    default: 'pending' // Transaction status
  },
  transactionId: { type: String, unique: true }, // Unique identifier for the transaction
  timestamp: { type: Date, default: Date.now }, // Timestamp of the transaction
  planet: { type: String, required: true }, // Planet where the transaction is initiated
  notes: { type: String }, // Optional notes for the transaction
});

// Method to update the transaction status
transactionSchema.methods.updateStatus = async function(newStatus) {
  this.status = newStatus;
  await this.save();
};

// Method to get a summary of the transaction
transactionSchema.methods.getSummary = function() {
  return {
    transactionId: this.transactionId,
    sender: this.sender,
    recipient: this.recipient,
    amount: this.amount,
    currency: this.currency,
    status: this.status,
    timestamp: this.timestamp,
    planet: this.planet,
    notes: this.notes,
  };
};

// Create the Transaction model
const Transaction = mongoose.model('Transaction', transactionSchema);

module.exports = Transaction;
