// Import required modules
const CurrencyExchange = require('./dbConfig').CurrencyExchange;
const { getExchangeRate, updateExchangeRates } = require('./exchangeRateConfig');

// Currency Controller
class CurrencyController {
  // Method to get the exchange rate for a specific currency pair
  static async getExchangeRate(req, res) {
    const { fromCurrency, toCurrency } = req.params;

    try {
      const rate = await getExchangeRate(fromCurrency, toCurrency);
      return res.status(200).json({ fromCurrency, toCurrency, exchangeRate: rate });
    } catch (error) {
      console.error('Error getting exchange rate:', error);
      return res.status(404).json({ message: error.message });
    }
  }

  // Method to update exchange rates in the database
  static async updateExchangeRates(req, res) {
    try {
      await updateExchangeRates();
      return res.status(200).json({ message: 'Exchange rates updated successfully' });
    } catch (error) {
      console.error('Error updating exchange rates:', error);
      return res.status(500).json({ message: 'Failed to update exchange rates' });
    }
  }

  // Method to get all available currency pairs and their rates
  static async getAllCurrencyPairs(req, res) {
    try {
      const rates = await CurrencyExchange.find({});
      return res.status(200).json(rates);
    } catch (error) {
      console.error('Error retrieving currency pairs:', error);
      return res.status(500).json({ message: 'Failed to retrieve currency pairs' });
    }
  }

  // Method to add a new currency exchange rate
  static async addCurrencyExchangeRate(req, res) {
    const { fromCurrency, toCurrency, exchangeRate } = req.body;

    try {
      const existingRate = await CurrencyExchange.findOne({ fromCurrency, toCurrency });
      if (existingRate) {
        return res.status(400).json({ message: 'Exchange rate already exists' });
      }

      const newRate = new CurrencyExchange({ fromCurrency, toCurrency, exchangeRate });
      await newRate.save();
      return res.status(201).json({ message: 'Exchange rate added successfully', newRate });
    } catch (error) {
      console.error('Error adding exchange rate:', error);
      return res.status(500).json({ message: 'Failed to add exchange rate' });
    }
  }

  // Method to delete a currency exchange rate
  static async deleteCurrencyExchangeRate(req, res) {
    const { fromCurrency, toCurrency } = req.params;

    try {
      const result = await CurrencyExchange.deleteOne({ fromCurrency, toCurrency });
      if (result.deletedCount === 0) {
        return res.status(404).json({ message: 'Exchange rate not found' });
      }
      return res.status(200).json({ message: 'Exchange rate deleted successfully' });
    } catch (error) {
      console.error('Error deleting exchange rate:', error);
      return res.status(500).json({ message: 'Failed to delete exchange rate' });
    }
  }
}

// Export the CurrencyController class
module.exports = CurrencyController;
