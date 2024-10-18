// src/currencies/currencyModel.js
const mongoose = require('mongoose');
const { isCurrencyCode, isCurrencyName, isCurrencySymbol } = require('../utils/validators');

const currencySchema = new mongoose.Schema({
    code: {
        type: String,
        required: true,
        unique: true,
        validate: {
            validator: isCurrencyCode,
            message: props => `${props.value} is not a valid currency code!`
        }
    },
    name: {
        type: String,
        required: true,
        validate: {
            validator: isCurrencyName,
            message: props => `${props.value} is not a valid currency name!`
        }
    },
    symbol: {
        type: String,
        required: true,
        validate: {
            validator: isCurrencySymbol,
            message: props => `${props.value} is not a valid currency symbol!`
        }
    },
    exchangeRate: {
        type: Number,
        required: true,
        min: 0
    },
    lastUpdated: {
        type: Date,
        default: Date.now
    }
});

// Method to update exchange rate
currencySchema.methods.updateExchangeRate = async function(newRate) {
    this.exchangeRate = newRate;
    this.lastUpdated = Date.now();
    await this.save();
};

// Static method to fetch and update exchange rates from an external API
currencySchema.statics.updateAllExchangeRates = async function() {
    const currencies = await this.find();
    const exchangeRates = await fetchExchangeRatesFromAPI(); // Hypothetical function to fetch rates

    for (const currency of currencies) {
        if (exchangeRates[currency.code]) {
            await currency.updateExchangeRate(exchangeRates[currency.code]);
        }
    }
};

// Hypothetical function to fetch exchange rates from an external API
async function fetchExchangeRatesFromAPI() {
    // Simulating an API call
    return {
        'USD': 1,
        'EUR': 0.85,
        'GBP': 0.75,
        'JPY': 110,
        // Add more currencies as needed
    };
}

module.exports = mongoose.model('Currency', currencySchema);
