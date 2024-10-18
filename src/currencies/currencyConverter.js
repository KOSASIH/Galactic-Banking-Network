// src/currencies/currencyConverter.js
const CurrencyService = require('./currencyService');
const logger = require('../utils/logger');

class CurrencyConverter {
    // Convert a single amount from one currency to another
    async convertSingle(amount, fromCurrencyCode, toCurrencyCode) {
        try {
            logger.info(`Converting ${amount} from ${fromCurrencyCode} to ${toCurrencyCode}`);
            const result = await CurrencyService.convertCurrency(amount, fromCurrencyCode, toCurrencyCode);
            if (!result.success) {
                throw new Error(result.message);
            }
            logger.info(`Conversion successful: ${amount} ${fromCurrencyCode} = ${result.convertedAmount} ${toCurrencyCode}`);
            return result.convertedAmount;
        } catch (error) {
            logger.error(`Conversion failed: ${error.message}`);
            throw new Error(`Conversion failed: ${error.message}`);
        }
    }

    // Convert multiple amounts from one currency to another
    async convertBatch(conversionRequests) {
        const results = [];
        for (const request of conversionRequests) {
            const { amount, fromCurrencyCode, toCurrencyCode } = request;
            try {
                const convertedAmount = await this.convertSingle(amount, fromCurrencyCode, toCurrencyCode);
                results.push({
                    amount,
                    fromCurrencyCode,
                    toCurrencyCode,
                    convertedAmount,
                    success: true
                });
            } catch (error) {
                results.push({
                    amount,
                    fromCurrencyCode,
                    toCurrencyCode,
                    success: false,
                    message: error.message
                });
            }
        }
        return results;
    }

    // Get exchange rate between two currencies
    async getExchangeRate(fromCurrencyCode, toCurrencyCode) {
        try {
            const fromCurrency = await CurrencyService.getCurrencyByCode(fromCurrencyCode);
            const toCurrency = await CurrencyService.getCurrencyByCode(toCurrencyCode);

            if (!fromCurrency.success || !toCurrency.success) {
                throw new Error('One or both currencies not found');
            }

            const exchangeRate = toCurrency.currency.exchangeRate / fromCurrency.currency.exchangeRate;
            return { success: true, exchangeRate };
        } catch (error) {
            logger.error(`Failed to get exchange rate: ${error.message}`);
            return { success: false, message: error.message };
        }
    }
}

module.exports = new CurrencyConverter();
