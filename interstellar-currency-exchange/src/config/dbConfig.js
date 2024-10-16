// Import required modules
const mongoose = require('mongoose');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// Define the database connection options
const dbOptions = {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  useCreateIndex: true,
  useFindAndModify: false,
  autoIndex: true,
  poolSize: 10,
  bufferMaxEntries: 0,
  connectTimeoutMS: 10000,
  socketTimeoutMS: 45000,
};

// Define the database connection URL
const dbUrl = process.env.MONGODB_URI;

// Create a mongoose connection
const connection = mongoose.createConnection(dbUrl, dbOptions);

// Define the database models
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: { type: String, required: true, enum: ['admin', 'user'] },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

const transactionSchema = new mongoose.Schema({
  fromCurrency: { type: String, required: true },
  toCurrency: { type: String, required: true },
  amount: { type: Number, required: true },
  convertedAmount: { type: Number, required: true },
  transactionDate: { type: Date, default: Date.now },
  transactionStatus: { type: String, required: true, enum: ['pending', 'success', 'failed'] },
});

const currencyExchangeSchema = new mongoose.Schema({
  fromCurrency: { type: String, required: true },
  toCurrency: { type: String, required: true },
  exchangeRate: { type: Number, required: true },
  updatedAt: { type: Date, default: Date.now },
});

// Create the database models
const User = connection.model('User ', userSchema);
const Transaction = connection.model('Transaction', transactionSchema);
const CurrencyExchange = connection.model('CurrencyExchange', currencyExchangeSchema);

// Export the database models
module.exports = { User, Transaction, CurrencyExchange };
