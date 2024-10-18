// tests/currencyService.test.js
const currencyService = require('../src/currency/currencyService');
const exchangeRateConfig = require('../config/exchangeRateConfig');
const axios = require('axios');

jest.mock('axios');

describe('Currency Service', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('should fetch exchange rates successfully', async () => {
        const mockResponse = {
            data: {
                rates: {
                    USD: 1,
                    EUR: 0.85,
                },
            },
        };

        axios.get.mockResolvedValue(mockResponse);

        const rates = await currencyService.getExchangeRates();
        expect(rates).toEqual(mockResponse.data.rates);
        expect(axios.get).toHaveBeenCalledWith(`${exchangeRateConfig.api.url}/${exchangeRateConfig.defaultCurrency}`);
    });

    test('should throw an error when fetching exchange rates fails', async () => {
        axios.get.mockRejectedValue(new Error('Network Error'));

        await expect(currencyService.getExchangeRates()).rejects.toThrow('Network Error');
    });

    test('should convert currency correctly', () => {
        const amount = 100;
        const fromRate = 1; // USD
        const toRate = 0.85; // EUR

        const convertedAmount = currencyService.convertCurrency(amount, fromRate, toRate);
        expect(convertedAmount).toBeCloseTo(85, 2);
    });
});
