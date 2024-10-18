// src/currencies/exchangeRateAPI.js
const axios = require('axios');
const NodeCache = require('node-cache');
const logger = require('../utils/logger');

// Initialize cache with a default TTL of 1 hour
const cache = new NodeCache({ stdTTL: 3600 });

class ExchangeRateAPI {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.apiUrl = 'https://api.exchangerate-api.com/v4/latest'; // Example API endpoint
    }

    // Fetch exchange rates from the API
    async fetchExchangeRates(baseCurrency = 'USD') {
        const cacheKey = `exchangeRates_${baseCurrency}`;
        
        // Check if rates are cached
        const cachedRates = cache.get(cacheKey);
        if (cachedRates) {
            logger.info(`Fetching exchange rates from cache for base currency: ${baseCurrency}`);
            return cachedRates;
        }

        try {
            logger.info(`Fetching exchange rates from API for base currency: ${baseCurrency}`);
            const response = await axios.get(`${this.apiUrl}/${baseCurrency}?apikey=${this.apiKey}`);
            const rates = response.data.rates;

            // Cache the rates
            cache.set(cacheKey, rates);
            return rates;
        } catch (error) {
            logger.error(`Failed to fetch exchange rates: ${error.message}`);
            throw new Error('Could not fetch exchange rates from the API');
        }
    }

    // Update exchange rates in the database
    async updateExchangeRatesInDatabase(CurrencyService) {
        try {
            const rates = await this.fetchExchangeRates();
            for (const [code, rate] of Object.entries(rates)) {
                await CurrencyService.updateCurrency(code, { exchangeRate: rate });
            }
            logger.info('Exchange rates updated successfully in the database');
        } catch (error) {
            logger.error(`Failed to update exchange rates in database: ${error.message}`);
        }
    }
}

module.exports = new ExchangeRateAPI(process.env.EXCHANGE_RATE_API_KEY); // Ensure to set your API key in environment variables
