// Import required modules
const express = require('express');
const app = express();
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const { errorHandler } = require('./errorHandler');
const { logger } = require('./logger');
const { apiConfig } = require('./apiConfig');
const { dbConfig } = require('./dbConfig');
const { exchangeRateConfig } = require('./exchangeRateConfig');
const { currencyController } = require('./currencyController');
const { transactionController } = require('./transactionController');
const { userController } = require('./userController');
const { currencyModel } = require('./currencyModel');
const { transactionModel } = require('./transactionModel');
const { userModel } = require('./userModel');
const { exchangeRateService } = require('./exchangeRateService');
const { notificationService } = require('./notificationService');
const { transactionService } = require('./transactionService');

// Set up middleware
app.use(cors());
app.use(helmet());
app.use(morgan('combined'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Set up routes
app.use('/api/currencies', currencyController);
app.use('/api/transactions', transactionController);
app.use('/api/users', userController);

// Set up error handling
app.use(errorHandler);

// Set up logging
app.use(logger);

// Connect to database
dbConfig.connect();

// Start the server
const port = apiConfig.port;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});

// Set up exchange rate service
exchangeRateService.start();

// Set up notification service
notificationService.start();

// Set up transaction service
transactionService.start();
