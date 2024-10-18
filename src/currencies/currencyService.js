// src/currencies/currencyService.js
const Currency = require('./currencyModel');
const { fetchExchangeRates } = require('./exchangeRateAPI');

class CurrencyService {
    // Add a new currency
    async addCurrency(currencyData) {
        try {
            const currency = new Currency(currencyData);
            await currency.save();
            return { success: true, currency };
        } catch (error) {
            return { success: false, message: this.handleError(error) };
        }
    }

    // Get all currencies
    async getAllCurrencies() {
        try {
            const currencies = await Currency.find();
            return { success: true, currencies };
        } catch (error) {
            return { success: false, message: this.handleError(error) };
        }
    }

    // Get currency by code
    async getCurrencyByCode(code) {
        try {
            const currency = await Currency.findOne({ code });
            if (!currency) throw new Error('Currency not found');
            return { success: true, currency };
        } catch (error) {
            return { success: false, message: this.handleError(error) };
        }
    }

    // Update an existing currency
    async updateCurrency(code, updateData) {
        try {
            const currency = await Currency.findOneAndUpdate({ code }, updateData, { new: true });
            if (!currency) throw new Error('Currency not found');
            return { success: true, currency };
        } catch (error) {
            return { success: false, message: this.handleError(error) };
        }
    }

    // Convert currency
    async convertCurrency(amount, fromCurrencyCode, toCurrencyCode) {
        try {
            const fromCurrency = await this.getCurrencyByCode(fromCurrencyCode);
            const toCurrency = await this.getCurrencyByCode(toCurrencyCode);

            if (!fromCurrency.success || !toCurrency.success) {
                throw new Error('One or both currencies not found');
            }

            const convertedAmount = (amount / fromCurrency.currency.exchangeRate) * toCurrency.currency.exchangeRate;
            return { success: true, convertedAmount };
        } catch (error) {
            return { success: false, message: this.handleError(error) };
        }
    }

    // Update all exchange rates from an external API
    async updateAllExchangeRates() {
        try {
            const exchangeRates = await fetchExchangeRates();
            for (const [code, rate] of Object.entries(exchangeRates)) {
                await Currency.updateOne({ code }, { exchangeRate: rate });
            }
            return { success: true, message: 'Exchange rates updated successfully' };
        } catch (error) {
            return { success: false, message: this.handleError(error) };
        }
    }

    // Handle errors
    handleError(error) {
        if (error.name === 'ValidationError') {
            return 'Validation error: ' + error.message;
        }
        return error.message || 'An unknown error occurred';
    }
}

module.exports = new CurrencyService();
