// index.js

const readline = require('readline');
const { compressGzip, compressDeflate, compressBrotli } = require('./data-compression/compression');
const { decompress, decompressGzip, decompressDeflate, decompressBrotli } = require('./data-compression/decompression');
const { storeData, retrieveData, deleteData, listStoredKeys, generateUniqueKey } = require('./data-storage');

// Create an interface for user input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

/**
 * Prompts the user for input and returns a promise
 * @param {string} question - The question to prompt the user
 * @returns {Promise<string>} - A promise that resolves to the user's input
 */
const askQuestion = (question) => {
    return new Promise((resolve) => {
        rl.question(question, (answer) => {
            resolve(answer);
        });
    });
};

/**
 * Main function to run the application
 */
const main = async () => {
    console.log("Welcome to the Galactic Data Management System!");
    
    while (true) {
        console.log("\nChoose an option:");
        console.log("1. Store Data");
        console.log("2. Retrieve Data");
        console.log("3. Delete Data");
        console.log("4. List Stored Keys");
        console.log("5. Compress Data");
        console.log("6. Decompress Data");
        console.log("7. Exit");

        const choice = await askQuestion("Enter your choice (1-7): ");

        switch (choice) {
            case '1':
                const keyToStore = await askQuestion("Enter a unique key for the data: ");
                const dataToStore = await askQuestion("Enter the data to store: ");
                await storeData(keyToStore, { data: dataToStore });
                break;

            case '2':
                const keyToRetrieve = await askQuestion("Enter the key of the data to retrieve: ");
                try {
                    const retrievedData = await retrieveData(keyToRetrieve);
                    console.log("Retrieved Data:", retrievedData);
                } catch (error) {
                    console.error(error.message);
                }
                break;

            case '3':
                const keyToDelete = await askQuestion("Enter the key of the data to delete: ");
                await deleteData(keyToDelete);
                break;

            case '4':
                const keys = await listStoredKeys();
                console.log("Stored Keys:", keys);
                break;

            case '5':
                const dataToCompress = await askQuestion("Enter the data to compress: ");
                const algorithm = await askQuestion("Choose compression algorithm (gzip, deflate, brotli): ");
                let compressedData;
                try {
                    switch (algorithm) {
                        case 'gzip':
                            compressedData = await compressGzip(dataToCompress);
                            break;
                        case 'deflate':
                            compressedData = await compressDeflate(dataToCompress);
                            break;
                        case 'brotli':
                            compressedData = await compressBrotli(dataToCompress);
                            break;
                        default:
                            console.error("Unsupported compression algorithm.");
                            continue;
                    }
                    console.log("Compressed Data:", compressedData.toString('base64')); // Display as base64 for readability
                } catch (error) {
                    console.error(error.message);
                }
                break;

            case '6':
                const keyToDecompress = await askQuestion("Enter the key of the compressed data: ");
                try {
                    const compressedData = await retrieveData(keyToDecompress);
                    const algorithmToDecompress = await askQuestion("Choose decompression algorithm (gzip, deflate, brotli): ");
                    let decompressedData;
                    switch (algorithmToDecompress) {
                        case 'gzip':
                            decompressedData = await decompressGzip(Buffer.from(compressedData.data, 'base64'));
                            break;
                        case 'deflate':
                            decompressedData = await decompressDeflate(Buffer.from(compressedData.data, 'base64'));
                            break;
                        case 'brotli':
                            decompressedData = await decompressBrotli(Buffer.from(compressedData.data, 'base64'));
                            break;
                        default:
                            console.error("Unsupported decompression algorithm.");
                            continue;
                    }
                    console.log("Decompressed Data:", decompressedData.toString());
                } catch (error) {
                    console.error(error.message);
                }
                break;

            case '7':
                console.log("Exiting the Galactic Data Management System. Goodbye!");
                rl.close();
                return;

            default :
                console.error("Invalid choice. Please choose a valid option.");
        }
    }
};

main();
