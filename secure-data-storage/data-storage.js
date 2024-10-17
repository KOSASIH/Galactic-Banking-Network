// data-storage.js

const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid'); // For generating unique IDs

const STORAGE_PATH = path.join(__dirname, 'data-storage'); // Directory for local storage

// Ensure the storage directory exists
const ensureStorageDirectory = async () => {
    try {
        await fs.mkdir(STORAGE_PATH, { recursive: true });
    } catch (error) {
        console.error('Failed to create storage directory:', error.message);
        throw new Error('Storage directory creation failed.');
    }
};

/**
 * Stores data in a JSON file
 * @param {string} key - The unique key for the data
 * @param {Object} data - The data to store
 * @returns {Promise<string>} - A promise that resolves to the storage path
 */
const storeData = async (key, data) => {
    await ensureStorageDirectory();
    const filePath = path.join(STORAGE_PATH, `${key}.json`);
    try {
        await fs.writeFile(filePath, JSON.stringify(data, null, 2));
        console.log(`Data stored successfully at ${filePath}`);
        return filePath;
    } catch (error) {
        console.error('Failed to store data:', error.message);
        throw new Error('Data storage failed.');
    }
};

/**
 * Retrieves data from a JSON file
 * @param {string} key - The unique key for the data
 * @returns {Promise<Object>} - A promise that resolves to the retrieved data
 */
const retrieveData = async (key) => {
    const filePath = path.join(STORAGE_PATH, `${key}.json`);
    try {
        const data = await fs.readFile(filePath, 'utf-8');
        console.log(`Data retrieved successfully from ${filePath}`);
        return JSON.parse(data);
    } catch (error) {
        console.error('Failed to retrieve data:', error.message);
        throw new Error('Data retrieval failed.');
    }
};

/**
 * Deletes data from a JSON file
 * @param {string} key - The unique key for the data
 * @returns {Promise<void>} - A promise that resolves when the data is deleted
 */
const deleteData = async (key) => {
    const filePath = path.join(STORAGE_PATH, `${key}.json`);
    try {
        await fs.unlink(filePath);
        console.log(`Data deleted successfully from ${filePath}`);
    } catch (error) {
        console.error('Failed to delete data:', error.message);
        throw new Error('Data deletion failed.');
    }
};

/**
 * Lists all stored data keys
 * @returns {Promise<Array<string>>} - A promise that resolves to an array of keys
 */
const listStoredKeys = async () => {
    try {
        const files = await fs.readdir(STORAGE_PATH);
        const keys = files.map(file => path.basename(file, '.json'));
        console.log('Stored keys:', keys);
        return keys;
    } catch (error) {
        console.error('Failed to list stored keys:', error.message);
        throw new Error('Key listing failed.');
    }
};

/**
 * Generates a unique key for data storage
 * @returns {string} - A unique key
 */
const generateUniqueKey = () => {
    return uuidv4();
};

// Exporting the functions
module.exports = {
    storeData,
    retrieveData,
    deleteData,
    listStoredKeys,
    generateUniqueKey,
};
