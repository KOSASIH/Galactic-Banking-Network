// src/utils/validators.js

function isCurrencyCode(code) {
    const currencyCodeRegex = /^[A-Z]{3}$/; // ISO 4217 format
    return currencyCodeRegex.test(code);
}

function isCurrencyName(name) {
    return typeof name === 'string' && name.length > 0;
}

function isCurrencySymbol(symbol) {
    return typeof symbol === 'string' && symbol.length > 0;
}

module.exports = {
    isCurrencyCode,
    isCurrencyName,
    isCurrencySymbol
};
