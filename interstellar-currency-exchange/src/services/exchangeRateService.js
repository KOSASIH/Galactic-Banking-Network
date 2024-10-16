// Import required modules
const axios = require('axios');
const moment = require('moment');
const { Currency } = require('./currencyModel');
const { CurrencyExchange } = require('./currencyExchangeModel');

// Define the ExchangeRateService class
class ExchangeRateService {
  // Method to get the latest exchange rates from an external API
  async getLatestExchangeRates() {
    try {
      const response = await axios.get('https://api.exchangeratesapi.io/latest');
      const data = response.data;
      const exchangeRates = data.rates;
      return exchangeRates;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  // Method to update the exchange rates in the database
  async updateExchangeRates(exchangeRates) {
    try {
      const currencies = await Currency.find();
      const currencyExchanges = [];

      for (const currency of currencies) {
        for (const exchangeRate of exchangeRates) {
          if (currency.code === exchangeRate.currency) {
            const currencyExchange = new CurrencyExchange({
              fromCurrency: currency.code,
              toCurrency: 'EUR',
              exchangeRate: exchangeRate.rate,
            });
            currencyExchanges.push(currencyExchange);
          }
        }
      }

      await CurrencyExchange.deleteMany({});
      await CurrencyExchange.insertMany(currencyExchanges);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  // Method to get the exchange rate for a specific currency pair
  async getExchangeRate(fromCurrency, toCurrency) {
    try {
      const currencyExchange = await CurrencyExchange.findOne({
        fromCurrency,
        toCurrency,
      });
      return currencyExchange.exchangeRate;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  // Method to convert an amount from one currency to another
  async convertAmount(amount, fromCurrency, toCurrency) {
    try {
      const exchangeRate = await this.getExchangeRate(fromCurrency, toCurrency);
      const convertedAmount = amount * exchangeRate;
      return convertedAmount;
    } catch (error) {
      console.error(error);
      return null;
    }
  }
}

// Export the ExchangeRateService class
module.exports = ExchangeRateService;
