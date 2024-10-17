// utils/constants.js

// API Configuration
const API_BASE_URL = process.env.API_BASE_URL || 'https://api.intergalactictransactions.com';
const API_TIMEOUT = process.env.API_TIMEOUT || 5000; // in milliseconds

// Notification Types
const NOTIFICATION_TYPES = {
    EMAIL: 'email',
    SMS: 'sms',
};

// Transaction Statuses
const TRANSACTION_STATUSES = {
    PENDING: 'pending',
    COMPLETED: 'completed',
    FAILED: 'failed',
    REFUNDED: 'refunded',
};

// Error Messages
const ERROR_MESSAGES = {
    TRANSACTION_NOT_FOUND: 'Transaction not found.',
    USER_NOT_AUTHORIZED: 'User  is not authorized to perform this action.',
    INVALID_INPUT: 'Invalid input provided.',
    NOTIFICATION_TYPE_UNSUPPORTED: 'Unsupported notification type.',
};

// Currency Constants
const CURRENCIES = {
    GALACTIC_CREDIT: 'GCR',
    INTERGALACTIC_DOLLAR: 'IGD',
};

// Miscellaneous Constants
const DEFAULT_PAGE_LIMIT = 10;
const DEFAULT_PAGE_NUMBER = 1;

module.exports = {
    API_BASE_URL,
    API_TIMEOUT,
    NOTIFICATION_TYPES,
    TRANSACTION_STATUSES,
    ERROR_MESSAGES,
    CURRENCIES,
    DEFAULT_PAGE_LIMIT,
    DEFAULT_PAGE_NUMBER,
};
