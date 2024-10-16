// Import required modules
const express = require('express');
const axios = require('axios');
const mongoose = require('mongoose');
const logger = require('../utils/logger');
const errorHandler = require('../utils/errorHandler');
const currencyConverter = require('../utils/currencyConverter');
const transactionProcessor = require('../utils/transactionProcessor');
const notificationService = require('../utils/notificationService');

// Define the currency exchange API
const currencyExchange = express.Router();

// Connect to the MongoDB database
mongoose.connect('mongodb://localhost/currency-exchange', {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

// Define the currency schema
const currencySchema = new mongoose.Schema({
  code: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  symbol: { type: String, required: true },
  exchangeRate: { type: Number, required: true },
  lastUpdated: { type: Date, default: Date.now }
});

// Create the currency model
const Currency = mongoose.model('Currency', currencySchema);

// Define the transaction schema
const transactionSchema = new mongoose.Schema({
  fromCurrency: { type: String, required: true },
  toCurrency: { type: String, required: true },
  amount: { type: Number, required: true },
  convertedAmount: { type: Number, required: true },
  transactionDate: { type: Date, default: Date.now }
});

// Create the transaction model
const Transaction = mongoose.model('Transaction', transactionSchema);

// Define the API endpoints
currencyExchange.get('/exchange-rates', async (req, res) => {
  try {
    // Fetch the latest exchange rates from the API
    const response = await axios.get('https://api.exchangerate-api.com/v4/latest/USD');
    const exchangeRates = response.data.rates;

    // Update the exchange rates in the database
    await Currency.updateMany({}, { exchangeRate: exchangeRates });

    // Return the updated exchange rates
    const currencies = await Currency.find();
    res.json(currencies);
  } catch (error) {
    errorHandler.handleError(error, res);
  }
});

currencyExchange.post('/convert', async (req, res) => {
  try {
    // Get the conversion details from the request body
    const { fromCurrency, toCurrency, amount } = req.body;

    // Validate the conversion details
    if (!fromCurrency || !toCurrency || !amount) {
      return res.status(400).json({ message: 'Invalid conversion details' });
    }

    // Fetch the exchange rates for the from and to currencies
    const fromCurrencyRate = await Currency.findOne({ code: fromCurrency });
    const toCurrencyRate = await Currency.findOne({ code: toCurrency });

    // Validate the exchange rates
    if (!fromCurrencyRate || !toCurrencyRate) {
      return res.status(404).json({ message: 'Currency not found' });
    }

    // Convert the amount using the exchange rates
    const convertedAmount = currencyConverter.convert(amount, fromCurrencyRate.exchangeRate, toCurrencyRate.exchangeRate);

    // Return the converted amount
    res.json({ convertedAmount });
  } catch (error) {
    errorHandler.handleError(error, res);
  }
});

currencyExchange.post('/transaction', async (req, res) => {
  try {
    // Get the transaction details from the request body
    const { fromCurrency, toCurrency, amount } = req.body;

    // Validate the transaction details
    if (!fromCurrency || !toCurrency || !amount) {
      return res.status(400).json({ message: 'Invalid transaction details' });
    }

    // Fetch the exchange rates for the from and to currencies
    const fromCurrencyRate = await Currency.findOne({ code: fromCurrency });
    const toCurrencyRate = await Currency.findOne({ code: toCurrency });

    // Validate the exchange rates
    if (!fromCurrencyRate || !toCurrencyRate) {
      return res.status(404).json({ message: 'Currency not found' });
    }

    // Convert the amount using the exchange rates
    const convertedAmount = currencyConverter.convert(amount, fromCurrencyRate.exchangeRate, toCurrencyRate.exchangeRate);

    // Process the transaction
    const transaction = await transactionProcessor.processTransaction(fromCurrency, toCurrency, amount, convertedAmount);

    // Save the transaction to the database
    await Transaction.create(transaction);

    // Send a notification to the user
    notificationService.sendNotification(`Transaction successful: ${amount} ${fromCurrency} -> ${convertedAmount} ${toCurrency}`);

    // Return the transaction result
    res.json({ message: 'Transaction processed successfully' });
  } catch (error) {
    errorHandler.handleError(error, res);
  }
});

// Export the currency exchange API
module.exports = currencyExchange;
