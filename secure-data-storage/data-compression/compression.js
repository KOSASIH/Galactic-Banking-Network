// data-compression/compression.js

const zlib = require('zlib');

/**
 * Compresses a string or buffer using Gzip
 * @param {string|Buffer} data - The data to compress
 * @returns {Promise<Buffer>} - A promise that resolves to the compressed data
 */
const compressGzip = (data) => {
    return new Promise((resolve, reject) => {
        zlib.gzip(data, (err, buffer) => {
            if (err) {
                console.error('Compression failed:', err.message);
                return reject(new Error('Compression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Decompresses Gzip compressed data
 * @param {Buffer} compressedData - The compressed data
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 */
const decompressGzip = (compressedData) => {
    return new Promise((resolve, reject) => {
        zlib.gunzip(compressedData, (err, buffer) => {
            if (err) {
                console.error('Decompression failed:', err.message);
                return reject(new Error('Decompression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Compresses a string or buffer using Deflate
 * @param {string|Buffer} data - The data to compress
 * @returns {Promise<Buffer>} - A promise that resolves to the compressed data
 */
const compressDeflate = (data) => {
    return new Promise((resolve, reject) => {
        zlib.deflate(data, (err, buffer) => {
            if (err) {
                console.error('Compression failed:', err.message);
                return reject(new Error('Compression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Decompresses Deflate compressed data
 * @param {Buffer} compressedData - The compressed data
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 */
const decompressDeflate = (compressedData) => {
    return new Promise((resolve, reject) => {
        zlib.inflate(compressedData, (err, buffer) => {
            if (err) {
                console.error('Decompression failed:', err.message);
                return reject(new Error('Decompression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Compresses data using Brotli
 * @param {string|Buffer} data - The data to compress
 * @returns {Promise<Buffer>} - A promise that resolves to the compressed data
 */
const compressBrotli = (data) => {
    return new Promise((resolve, reject) => {
        zlib.brotliCompress(data, (err, buffer) => {
            if (err) {
                console.error('Compression failed:', err.message);
                return reject(new Error('Compression failed.'));
            }
            resolve(buffer);
        });
    });
};

/**
 * Decompresses Brotli compressed data
 * @param {Buffer} compressedData - The compressed data
 * @returns {Promise<string|Buffer>} - A promise that resolves to the decompressed data
 */
const decompressBrotli = (compressedData) => {
    return new Promise((resolve, reject) => {
        zlib.brotliDecompress(compressedData, (err, buffer) => {
            if (err) {
                console.error('Decompression failed:', err.message);
                return reject(new Error('Decompression failed.'));
            }
            resolve(buffer);
        });
    });
};

// Exporting the functions
module.exports = {
    compressGzip,
    decompressGzip,
    compressDeflate,
    decompressDeflate,
    compressBrotli,
    decompressBrotli,
};
