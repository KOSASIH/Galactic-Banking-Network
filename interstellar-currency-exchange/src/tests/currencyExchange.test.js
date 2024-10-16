// Import required modules
const { CurrencyExchange } = require('./currencyExchangeModel');
const { ExchangeRateService } = require('./exchangeRateService');
const mongoose = require('mongoose');

// Define the test suite for CurrencyExchange
describe('CurrencyExchange', () => {
  // Before each test, connect to the database and create a new exchange rate service
  beforeEach(async () => {
    await mongoose.connect('mongodb://localhost/test', { useNewUrlParser: true, useUnifiedTopology: true });
    this.exchangeRateService = new ExchangeRateService();
  });

  // After each test, disconnect from the database
  afterEach(async () => {
    await mongoose.disconnect();
  });

  // Test 1: Create a new currency exchange
  it('should create a new currency exchange', async () => {
    const fromCurrency = 'USD';
    const toCurrency = 'EUR';
    const exchangeRate = 0.88;

    const currencyExchange = new CurrencyExchange({
      fromCurrency,
      toCurrency,
      exchangeRate,
    });

    await currencyExchange.save();

    expect(currencyExchange.fromCurrency).toBe(fromCurrency);
    expect(currencyExchange.toCurrency).toBe(toCurrency);
    expect(currencyExchange.exchangeRate).toBeCloseTo(exchangeRate, 2);
  });

  // Test 2: Get all currency exchanges
  it('should get all currency exchanges', async () => {
    const currencyExchanges = await CurrencyExchange.find();

    expect(currencyExchanges).toBeInstanceOf(Array);
    expect(currencyExchanges.length).toBeGreaterThan(0);
  });

  // Test 3: Get a currency exchange by ID
  it('should get a currency exchange by ID', async () => {
    const currencyExchange = await CurrencyExchange.create({
      fromCurrency: 'USD',
      toCurrency: 'EUR',
      exchangeRate: 0.88,
    });

    const retrievedCurrencyExchange = await CurrencyExchange.findById(currencyExchange.id);

    expect(retrievedCurrencyExchange.fromCurrency).toBe(currencyExchange.fromCurrency);
    expect(retrievedCurrencyExchange.toCurrency).toBe(currencyExchange.toCurrency);
    expect(retrievedCurrencyExchange.exchangeRate).toBeCloseTo(currencyExchange.exchangeRate, 2);
  });

  // Test 4: Update a currency exchange
  it('should update a currency exchange', async () => {
    const currencyExchange = await CurrencyExchange.create({
      fromCurrency: 'USD',
      toCurrency: 'EUR',
      exchangeRate: 0.88,
    });

    currencyExchange.exchangeRate = 0.90;
    await currencyExchange.save();

    const retrievedCurrencyExchange = await CurrencyExchange.findById(currencyExchange.id);

    expect(retrievedCurrencyExchange.exchangeRate).toBeCloseTo(0.90, 2);
  });

  // Test 5: Delete a currency exchange
  it('should delete a currency exchange', async () => {
    const currencyExchange = await CurrencyExchange.create({
      fromCurrency: 'USD',
      toCurrency: 'EUR',
      exchangeRate: 0.88,
    });

    await currencyExchange.remove();

    const retrievedCurrencyExchange = await CurrencyExchange.findById(currencyExchange.id);

    expect(retrievedCurrencyExchange).toBeNull();
  });

  // Test 6: Get exchange rate using ExchangeRateService
  it('should get exchange rate using ExchangeRateService', async () => {
    const fromCurrency = 'USD';
    const toCurrency = 'EUR';

    const exchangeRate = await this.exchangeRateService.getExchangeRate(fromCurrency, toCurrency);

    expect(exchangeRate).toBeGreaterThan(0);
  });
});
