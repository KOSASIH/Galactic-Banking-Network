// config/exchangeRateConfig.js
const exchangeRateConfig = {
    api: {
        url: process.env.EXCHANGE_RATE_API_URL || 'https://api.exchangerate-api.com/v4/latest',
        key: process.env.EXCHANGE_RATE_API_KEY || 'your_api_key',
    },
    defaultCurrency: process.env.DEFAULT_CURRENCY || 'USD',
    updateInterval: process.env.UPDATE_INTERVAL || 3600, // in seconds
};

module.exports = exchangeRateConfig;
