// tests/database/TransactionRepository.test.js

const { TransactionRepository } = require('../../database/TransactionRepository');
const { TRANSACTION_STATUSES } = require('../../utils/constants');

describe('TransactionRepository', () => {
    it('should create a new transaction', async () => {
        const transactionRepository = new TransactionRepository();
        const transaction = {
            amount: 100,
            currency: 'GCR',
            planet: 'Earth',
        };

        const createdTransaction = await transactionRepository.createTransaction(transaction);

        expect(createdTransaction.status).toBe(TRANSACTION_STATUSES.PENDING);
    });

    it('should retrieve a transaction by ID', async () => {
        const transactionRepository = new TransactionRepository();
        const transactionId = '1234567890';
        const transaction = {
            id: transactionId,
            amount: 100,
            currency: 'GCR',
            planet: 'Earth',
        };

        await transactionRepository.createTransaction(transaction);
        const retrievedTransaction = await transactionRepository.getTransaction(transactionId);

        expect(retrievedTransaction.id).toBe(transactionId);
    });

    it('should update a transaction status', async () => {
        const transactionRepository = new TransactionRepository();
        const transactionId = '1234567890';
        const transaction = {
            id: transactionId,
            amount: 100,
            currency: 'GCR',
            planet: 'Earth',
        };

        await transactionRepository.createTransaction(transaction);
        await transactionRepository.updateTransactionStatus(transactionId, TRANSACTION_STATUSES.COMPLETED);

        const updatedTransaction = await transactionRepository.getTransaction(transactionId);

        expect(updatedTransaction.status).toBe(TRANSACTION_STATUSES.COMPLETED);
    });
});
