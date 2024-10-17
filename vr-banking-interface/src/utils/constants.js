// src/utils/constants.js

// VR Mode Constants
export const VR_MODE = {
    ENABLED: 'ENABLED',
    DISABLED: 'DISABLED',
};

// API Endpoints
export const API_ENDPOINTS = {
    TRANSACTIONS: '/api/transactions',
    ACCOUNT_DETAILS: '/api/account',
    VR_ACTIONS: '/api/vr',
};

// Transaction Types
export const TRANSACTION_TYPES = {
    DEPOSIT: 'Deposit',
    WITHDRAWAL: 'Withdrawal',
    TRANSFER: 'Transfer',
};

// Error Messages
export const ERROR_MESSAGES = {
    FETCH_TRANSACTIONS: 'Failed to fetch transactions. Please try again later.',
    FETCH_ACCOUNT_DETAILS: 'Failed to fetch account details. Please try again later.',
    VR_NOT_SUPPORTED: 'WebVR is not supported on this device.',
    VR_NO_DISPLAYS: 'No VR displays found.',
};

// User Roles
export const USER_ROLES = {
    ADMIN: 'ADMIN',
    USER: 'USER',
    GUEST: 'GUEST',
};

// Currency Constants
export const CURRENCY = {
    SYMBOL: 'GCR',
    NAME: 'Global Currency',
};

// Miscellaneous Constants
export const APP_NAME = 'Virtual Reality Banking Interface';
export const APP_VERSION = '1.0.0';
