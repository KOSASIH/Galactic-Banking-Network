// Import required modules
const winston = require('winston');
const { format } = require('winston');
const { combine, timestamp, label, printf } = format;
const DailyRotateFile = require('winston-daily-rotate-file');

// Define the custom log format
const customFormat = printf(({ level, message, label, timestamp }) => {
  return `${timestamp} [${label}] ${level}: ${message}`;
});

// Create a new logger instance
const logger = winston.createLogger({
  level: 'info',
  format: combine(
    label({ label: 'Logger' }),
    timestamp(),
    customFormat
  ),
  transports: [
    new winston.transports.Console(),
    new DailyRotateFile({
      filename: 'logs/%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      zippedArchive: true,
      maxSize: '20m',
      maxFiles: '14d',
    }),
  ],
});

// Define the log levels and their corresponding methods
const logLevels = {
  error: logger.error,
  warn: logger.warn,
  info: logger.info,
  debug: logger.debug,
  verbose: logger.verbose,
  silly: logger.silly,
};

// Export the logger instance and log levels
module.exports = {
  logger,
  logLevels,
};
