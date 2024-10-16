// Import required modules
const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Define the Currency schema
const currencySchema = new Schema({
  _id: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  symbol: { type: String, required: true },
  exchangeRate: { type: Number, required: true },
  lastUpdated: { type: Date, default: Date.now },
});

// Define the Currency model
const Currency = mongoose.model('Currency', currencySchema);

// Export the Currency model
module.exports = Currency;

// Define the CurrencyExchange schema
const currencyExchangeSchema = new Schema({
  fromCurrency: { type: String, required: true },
  toCurrency: { type: String, required: true },
  exchangeRate: { type: Number, required: true },
  lastUpdated: { type: Date, default: Date.now },
});

// Define the CurrencyExchange model
const CurrencyExchange = mongoose.model('CurrencyExchange', currencyExchangeSchema);

// Export the CurrencyExchange model
module.exports = { Currency, CurrencyExchange };

// Define the Transaction schema
const transactionSchema = new Schema({
  fromCurrency: { type: String, required: true },
  toCurrency: { type: String, required: true },
  amount: { type: Number, required: true },
  convertedAmount: { type: Number, required: true },
  transactionDate: { type: Date, default: Date.now },
  transactionStatus: { type: String, required: true },
});

// Define the Transaction model
const Transaction = mongoose.model('Transaction', transactionSchema);

// Export the Transaction model
module.exports = { Currency, CurrencyExchange, Transaction };

// Define the User schema
const userSchema = new Schema({
  _id: { type: String, required: true, unique: true },
  username: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  lastLogin: { type: Date, default: Date.now },
});

// Define the User model
const User = mongoose.model('User ', userSchema);

// Export the User model
module.exports = { Currency, CurrencyExchange, Transaction, User };
