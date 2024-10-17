// tests/api/TransactionController.test.js

const request = require('supertest');
const app = require('../../app');
const { TRANSACTION_STATUSES } = require('../../utils/constants');

describe('TransactionController', () => {
    it('should create a new transaction', async () => {
        const response = await request(app)
            .post('/api/transactions')
            .send({
                amount: 100,
                currency: 'GCR',
                planet: 'Earth',
            });

        expect(response.status).toBe(201);
        expect(response.body.status).toBe(TRANSACTION_STATUSES.PENDING);
    });

    it('should retrieve a transaction by ID', async () => {
        const transactionId = '1234567890';
        const response = await request(app)
            .get(`/api/transactions/${transactionId}`);

        expect(response.status).toBe(200);
        expect(response.body.id).toBe(transactionId);
    });

    it('should update a transaction status', async () => {
        const transactionId = '1234567890';
        const response = await request(app)
            .patch(`/api/transactions/${transactionId}`)
            .send({
                status: TRANSACTION_STATUSES.COMPLETED,
            });

        expect(response.status).toBe(200);
        expect(response.body.status).toBe(TRANSACTION_STATUSES.COMPLETED);
    });

    it('should delete a transaction', async () => {
        const transactionId = '1234567890';
        const response = await request(app)
            .delete(`/api/transactions/${transactionId}`);

        expect(response.status).toBe(204);
    });
});
