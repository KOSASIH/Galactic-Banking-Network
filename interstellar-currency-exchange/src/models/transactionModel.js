// Import required modules
const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Define the Transaction schema
const transactionSchema = new Schema({
  _id: { type: String, required: true, unique: true },
  fromCurrency: { type: String, required: true, ref: 'Currency' },
  toCurrency: { type: String, required: true, ref: 'Currency' },
  amount: { type: Number, required: true },
  convertedAmount: { type: Number, required: true },
  transactionDate: { type: Date, default: Date.now },
  transactionStatus: { type: String, required: true, enum: ['pending', 'completed', 'failed'] },
  userId: { type: String, required: true, ref: 'User ' },
  exchangeRate: { type: Number, required: true },
  fees: { type: Number, required: true },
  totalAmount: { type: Number, required: true },
});

// Define the Transaction model
const Transaction = mongoose.model('Transaction', transactionSchema);

// Export the Transaction model
module.exports = Transaction;

// Define the Transaction methods
Transaction.methods = {
  // Method to calculate the converted amount
  calculateConvertedAmount: function() {
    return this.amount * this.exchangeRate;
  },

  // Method to calculate the total amount
  calculateTotalAmount: function() {
    return this.convertedAmount + this.fees;
  },

  // Method to update the transaction status
  updateTransactionStatus: function(status) {
    this.transactionStatus = status;
    return this.save();
  },
};

// Define the Transaction static methods
Transaction.statics = {
  // Method to get all transactions for a user
  getTransactionsBy:User  function(userId) {
    return this.find({ userId });
  },

  // Method to get a transaction by ID
  getTransactionById: function(transactionId) {
    return this.findById(transactionId);
  },
};

// Define the Transaction hooks
Transaction.pre('save', function(next) {
  // Calculate the converted amount and total amount before saving
  this.convertedAmount = this.calculateConvertedAmount();
  this.totalAmount = this.calculateTotalAmount();
  next();
});

Transaction.post('save', function(doc) {
  // Send a notification to the user after saving
  console.log(`Transaction saved: ${doc._id}`);
});

// Export the Transaction model with methods and static methods
module.exports = Transaction;
