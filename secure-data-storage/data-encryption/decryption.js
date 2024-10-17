// data-encryption/decryption.js

const crypto = require('crypto');
const { ALGORITHM, IV_LENGTH } = require('./constants'); // Import constants for algorithm and IV length

/**
 * Decrypts the given encrypted text using the specified algorithm
 * @param {string} encryptedText - The encrypted text in Base64 format
 * @param {Buffer} key - The decryption key
 * @returns {string} - The decrypted text
 * @throws Will throw an error if decryption fails
 */
const decrypt = (encryptedText, key) => {
    try {
        const parts = encryptedText.split(':');
        const iv = Buffer.from(parts.shift(), 'base64'); // Extract the IV
        const encryptedTextBuffer = Buffer.from(parts.join(':'), 'base64');
        const decipher = crypto.createDecipheriv(ALGORITHM, Buffer.from(key), iv);
        let decrypted = decipher.update(encryptedTextBuffer, 'base64', 'utf8');
        decrypted += decipher.final('utf8');
        return decrypted;
    } catch (error) {
        console.error('Decryption failed:', error.message);
        throw new Error('Decryption failed. Please check the encrypted text and key.');
    }
};

/**
 * Decrypts multiple encrypted texts in an array
 * @param {Array<string>} encryptedTexts - An array of encrypted texts
 * @param {Buffer} key - The decryption key
 * @returns {Array<string>} - An array of decrypted texts
 */
const decryptMultiple = (encryptedTexts, key) => {
    return encryptedTexts.map(encryptedText => decrypt(encryptedText, key));
};

/**
 * Validates the decryption key
 * @param {Buffer} key - The key to validate
 * @returns {boolean} - True if the key is valid, false otherwise
 */
const validateKey = (key) => {
    return Buffer.isBuffer(key) && key.length === 32; // Check if key is a Buffer and 256 bits
};

// Exporting the functions
module.exports = {
    decrypt,
    decryptMultiple,
    validateKey,
};
