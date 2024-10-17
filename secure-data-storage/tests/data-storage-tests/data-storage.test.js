// tests/data-storage-tests/data-storage.test.js

const fs = require('fs').promises;
const path = require('path');
const { storeData, retrieveData, deleteData, listStoredKeys, generateUniqueKey } = require('../../data-storage');

const STORAGE_PATH = path.join(__dirname, '../../data-storage');

beforeAll(async () => {
    // Ensure the storage directory exists before running tests
    await fs.mkdir(STORAGE_PATH, { recursive: true });
});

afterAll(async () => {
    // Clean up the storage directory after tests
    const files = await fs.readdir(STORAGE_PATH);
    await Promise.all(files.map(file => fs.unlink(path.join(STORAGE_PATH, file))));
});

describe('Data Storage Module', () => {
    test('should store data successfully', async () => {
        const key = generateUniqueKey();
        const data = { message: 'Hello, Galactic!' };
        const filePath = await storeData(key, data);
        
        expect(filePath).toBe(path.join(STORAGE_PATH, `${key}.json`));
        const storedData = await fs.readFile(filePath, 'utf-8');
        expect(JSON.parse(storedData)).toEqual(data);
    });

    test('should retrieve data successfully', async () => {
        const key = generateUniqueKey();
        const data = { message: 'Hello, Galactic!' };
        await storeData(key, data);
        
        const retrievedData = await retrieveData(key);
        expect(retrievedData).toEqual(data);
    });

    test('should delete data successfully', async () => {
        const key = generateUniqueKey();
        const data = { message: 'Hello, Galactic!' };
        await storeData(key, data);
        
        await deleteData(key);
        await expect(retrieveData(key)).rejects.toThrow('Data retrieval failed.');
    });

    test('should list stored keys successfully', async () => {
        const keys = [generateUniqueKey(), generateUniqueKey(), generateUniqueKey()];
        await Promise.all(keys.map(key => storeData(key, { message: 'Test' })));
        
        const storedKeys = await listStoredKeys();
        expect(storedKeys).toEqual(expect.arrayContaining(keys));
    });
});
