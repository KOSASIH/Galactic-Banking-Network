// data-compression/decompression.js

const zlib = require('zlib');

/**
 * Decompresses Gzip compressed data
 * @param {Buffer} compressedData - The compressed data
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 * @throws Will throw an error if decompression fails
 */
const decompressGzip = (compressedData) => {
    return new Promise((resolve, reject) => {
        zlib.gunzip(compressedData, (err, buffer) => {
            if (err) {
                console.error('Gzip Decompression failed:', err.message);
                return reject(new Error('Gzip Decompression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Decompresses Deflate compressed data
 * @param {Buffer} compressedData - The compressed data
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 * @throws Will throw an error if decompression fails
 */
const decompressDeflate = (compressedData) => {
    return new Promise((resolve, reject) => {
        zlib.inflate(compressedData, (err, buffer) => {
            if (err) {
                console.error('Deflate Decompression failed:', err.message);
                return reject(new Error('Deflate Decompression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Decompresses Brotli compressed data
 * @param {Buffer} compressedData - The compressed data
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 * @throws Will throw an error if decompression fails
 */
const decompressBrotli = (compressedData) => {
    return new Promise((resolve, reject) => {
        zlib.brotliDecompress(compressedData, (err, buffer) => {
            if (err) {
                console.error('Brotli Decompression failed:', err.message);
                return reject(new Error('Brotli Decompression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Decompresses data based on the specified algorithm
 * @param {Buffer} compressedData - The compressed data
 * @param {string} algorithm - The algorithm used for compression ('gzip', 'deflate', 'brotli')
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 * @throws Will throw an error if the algorithm is unsupported
 */
const decompress = (compressedData, algorithm) => {
    switch (algorithm) {
        case 'gzip':
            return decompressGzip(compressedData);
        case 'deflate':
            return decompressDeflate(compressedData);
        case 'brotli':
            return decompressBrotli(compressedData);
        default:
            return Promise.reject(new Error('Unsupported decompression algorithm.'));
    }
};

// Exporting the functions
module.exports = {
    decompressGzip,
    decompressDeflate,
    decompressBrotli,
    decompress,
};
