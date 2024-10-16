// Import required modules
const winston = require('winston');
const { format } = require('winston');
const { combine, timestamp, label, printf } = format;

// Define the custom log format
const customFormat = printf(({ level, message, label, timestamp }) => {
  return `${timestamp} [${label}] ${level}: ${message}`;
});

// Create a new logger instance
const logger = winston.createLogger({
  level: 'info',
  format: combine(
    label({ label: 'Error Handler' }),
    timestamp(),
    customFormat
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
  ],
});

// Define the error handler function
const errorHandler = (err, req, res, next) => {
  // Log the error using the custom logger
  logger.error(err.message);

  // Return a JSON response with the error message
  res.status(500).json({
    message: 'Internal Server Error',
    error: err.message,
  });
};

// Define the notFound handler function
const notFoundHandler = (req, res, next) => {
  // Log the not found error using the custom logger
  logger.info(`Route not found: ${req.url}`);

  // Return a JSON response with the not found message
  res.status(404).json({
    message: 'Route Not Found',
  });
};

// Export the error handler and not found handler functions
module.exports = {
  errorHandler,
  notFoundHandler,
};
