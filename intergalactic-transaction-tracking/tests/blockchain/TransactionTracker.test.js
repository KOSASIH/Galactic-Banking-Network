// tests/blockchain/TransactionTracker.test.js

const { TransactionTracker } = require('../../blockchain/TransactionTracker');
const { TRANSACTION_STATUSES } = require('../../utils/constants');

describe('TransactionTracker', () => {
    it('should track a new transaction', async () => {
        const transactionTracker = new TransactionTracker();
        const transactionId = '1234567890';
        const transaction = {
            id: transactionId,
            amount: 100,
            currency: 'GCR',
            planet: 'Earth',
        };

        await transactionTracker.trackTransaction(transaction);

        expect(transactionTracker.getTransaction(transactionId).status).toBe(TRANSACTION_STATUSES.PENDING);
    });

    it('should update a transaction status', async () => {
        const transactionTracker = new TransactionTracker();
        const transactionId = '1234567890';
        const transaction = {
            id: transactionId,
            amount: 100,
            currency: 'GCR',
            planet: 'Earth',
        };

        await transactionTracker.trackTransaction(transaction);
        await transactionTracker.updateTransactionStatus(transactionId, TRANSACTION_STATUSES.COMPLETED);

        expect(transactionTracker.getTransaction(transactionId).status).toBe(TRANSACTION_STATUSES.COMPLETED);
    });
});
