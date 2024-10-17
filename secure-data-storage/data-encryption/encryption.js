// data-encryption/encryption.js

const crypto = require('crypto');

// Constants for encryption
const ALGORITHM = 'aes-256-cbc';
const IV_LENGTH = 16; // Initialization vector length

/**
 * Generates a random key for encryption
 * @returns {Buffer} - A random key
 */
const generateKey = () => {
    return crypto.randomBytes(32); // 256 bits
};

/**
 * Encrypts the given text using AES-256-CBC
 * @param {string} text - The text to encrypt
 * @param {Buffer} key - The encryption key
 * @returns {string} - The encrypted text in Base64 format
 */
const encrypt = (text, key) => {
    const iv = crypto.randomBytes(IV_LENGTH); // Generate a random initialization vector
    const cipher = crypto.createCipheriv(ALGORITHM, Buffer.from(key), iv);
    let encrypted = cipher.update(text, 'utf8', 'base64');
    encrypted += cipher.final('base64');
    // Return the IV and encrypted text concatenated
    return `${iv.toString('base64')}:${encrypted}`;
};

/**
 * Decrypts the given encrypted text using AES-256-CBC
 * @param {string} encryptedText - The encrypted text in Base64 format
 * @param {Buffer} key - The decryption key
 * @returns {string} - The decrypted text
 */
const decrypt = (encryptedText, key) => {
    const parts = encryptedText.split(':');
    const iv = Buffer.from(parts.shift(), 'base64'); // Extract the IV
    const encryptedTextBuffer = Buffer.from(parts.join(':'), 'base64');
    const decipher = crypto.createDecipheriv(ALGORITHM, Buffer.from(key), iv);
    let decrypted = decipher.update(encryptedTextBuffer, 'base64', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
};

/**
 * Encodes data to Base64 format
 * @param {Buffer} data - The data to encode
 * @returns {string} - The Base64 encoded string
 */
const encodeBase64 = (data) => {
    return data.toString('base64');
};

/**
 * Decodes data from Base64 format
 * @param {string} base64String - The Base64 encoded string
 * @returns {Buffer} - The decoded data
 */
const decodeBase64 = (base64String) => {
    return Buffer.from(base64String, 'base64');
};

// Exporting the functions
module.exports = {
    generateKey,
    encrypt,
    decrypt,
    encodeBase64,
    decodeBase64,
};
