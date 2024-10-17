// config.js

// Load environment variables from a .env file
require('dotenv').config();

const config = {
    // Application settings
    app: {
        name: process.env.APP_NAME || 'Galactic Banking Network',
        port: process.env.PORT || 3000,
        environment: process.env.NODE_ENV || 'development',
    },
    // Blockchain settings
    blockchain: {
        network: process.env.BLOCKCHAIN_NETWORK || 'mainnet',
        providerUrl: process.env.PROVIDER_URL || 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID',
        contractAddress: process.env.CONTRACT_ADDRESS || '0xYourContractAddress',
    },
    // Database settings
    database: {
        host: process.env.DB_HOST || 'localhost',
        port: process.env.DB_PORT || 5432,
        user: process.env.DB_USER || 'user',
        password: process.env.DB_PASSWORD || 'password',
        name: process.env.DB_NAME || 'galactic_banking',
    },
    // API settings
    api: {
        key: process.env.API_KEY || 'your-api-key',
        endpoint: process.env.API_ENDPOINT || 'https://api.example.com',
    },
    // Logging settings
    logging: {
        level: process.env.LOG_LEVEL || 'info',
        logFile: process.env.LOG_FILE || 'app.log',
    },
};

module.exports = config;
