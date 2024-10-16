// Import required modules
const axios = require('axios');

// CurrencyConverter class to handle currency conversion
class CurrencyConverter {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.apiUrl = 'https://api.exchangerate-api.com/v4/latest/';
  }

  // Method to fetch exchange rates for a given base currency
  async fetchExchangeRates(baseCurrency) {
    try {
      const response = await axios.get(`${this.apiUrl}${baseCurrency}`);
      return response.data.rates;
    } catch (error) {
      throw new Error('Error fetching exchange rates: ' + error.message);
    }
  }

  // Method to convert an amount from one currency to another
  async convertCurrency(amount, fromCurrency, toCurrency) {
    if (amount <= 0) {
      throw new Error('Amount must be greater than zero');
    }

    const rates = await this.fetchExchangeRates(fromCurrency);
    
    if (!rates[toCurrency]) {
      throw new Error(`Currency ${toCurrency} not found`);
    }

    const convertedAmount = (amount * rates[toCurrency]).toFixed(2);
    return {
      fromCurrency,
      toCurrency,
      amount,
      convertedAmount,
    };
  }
}

// Example usage
(async () => {
  const converter = new CurrencyConverter('YOUR_API_KEY');

  try {
    const result = await converter.convertCurrency(100, 'USD', 'EUR');
    console.log(`Converted ${result.amount} ${result.fromCurrency} to ${result.convertedAmount} ${result.toCurrency}`);
  } catch (error) {
    console.error(error.message);
  }
})();

module.exports = CurrencyConverter;
