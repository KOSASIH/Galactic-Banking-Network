// tests/transactionService.test.js
const transactionService = require('../src/transaction/transactionService');
const Transaction = require('../src/transaction/transactionModel');

jest.mock('../src/transaction/transactionModel');

describe('Transaction Service', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('should create a transaction successfully', async () => {
        const mockTransaction = {
            amount: 100,
            fromCurrency: 'USD',
            toCurrency: 'EUR',
            convertedAmount: 85,
        };

        Transaction.prototype.save.mockResolvedValue(mockTransaction);

        const transaction = await transactionService.createTransaction(mockTransaction);
        expect(transaction).toEqual(mockTransaction);
        expect(Transaction.prototype.save).toHaveBeenCalled();
    });

    test('should throw an error when creating a transaction fails', async () => {
        Transaction.prototype.save.mockRejectedValue(new Error('Database Error'));

        await expect(transactionService.createTransaction({})).rejects.toThrow('Database Error');
    });

    test('should retrieve all transactions', async () => {
        const mockTransactions = [
            { amount: 100, fromCurrency: 'USD', toCurrency: 'EUR', convertedAmount: 85 },
            { amount: 200, fromCurrency: 'EUR', toCurrency: 'USD', convertedAmount: 235 },
        ];

        Transaction.find.mockResolvedValue(mockTransactions);

        const transactions = await transactionService.getAllTransactions();
        expect(transactions).toEqual(mockTransactions);
        expect(Transaction.find).toHaveBeenCalled();
    });
});
