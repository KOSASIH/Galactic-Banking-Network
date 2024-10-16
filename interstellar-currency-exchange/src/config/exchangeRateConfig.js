// Import required modules
const axios = require('axios');
const dotenv = require('dotenv');
const mongoose = require('mongoose');
const CurrencyExchange = require('./dbConfig').CurrencyExchange; // Import the CurrencyExchange model

// Load environment variables from .env file
dotenv.config();

// Define the API endpoint for fetching exchange rates
const EXCHANGE_RATE_API_URL = process.env.EXCHANGE_RATE_API_URL;

// Function to fetch exchange rates from an external API
const fetchExchangeRates = async () => {
  try {
    const response = await axios.get(EXCHANGE_RATE_API_URL);
    return response.data;
  } catch (error) {
    console.error('Error fetching exchange rates:', error);
    throw new Error('Failed to fetch exchange rates');
  }
};

// Function to update exchange rates in the database
const updateExchangeRates = async () => {
  try {
    const rates = await fetchExchangeRates();

    // Loop through the rates and update the database
    for (const [currencyPair, rate] of Object.entries(rates)) {
      const [fromCurrency, toCurrency] = currencyPair.split('/');
      const existingRate = await CurrencyExchange.findOne({ fromCurrency, toCurrency });

      if (existingRate) {
        // Update existing rate
        existingRate.exchangeRate = rate;
        existingRate.updatedAt = new Date();
        await existingRate.save();
      } else {
        // Create a new rate entry
        const newRate = new CurrencyExchange({
          fromCurrency,
          toCurrency,
          exchangeRate: rate,
        });
        await newRate.save();
      }
    }

    console.log('Exchange rates updated successfully');
  } catch (error) {
    console.error('Error updating exchange rates:', error);
  }
};

// Function to get the exchange rate for a specific currency pair
const getExchangeRate = async (fromCurrency, toCurrency) => {
  try {
    const rate = await CurrencyExchange.findOne({ fromCurrency, toCurrency });
    if (!rate) {
      throw new Error('Exchange rate not found');
    }
    return rate.exchangeRate;
  } catch (error) {
    console.error('Error retrieving exchange rate:', error);
    throw error;
  }
};

// Function to schedule regular updates of exchange rates
const scheduleExchangeRateUpdates = (interval) => {
  setInterval(() => {
    updateExchangeRates();
  }, interval);
};

// Export the functions
module.exports = {
  fetchExchangeRates,
  updateExchangeRates,
  getExchangeRate,
  scheduleExchangeRateUpdates,
};
