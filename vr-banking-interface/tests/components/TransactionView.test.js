// tests/components/TransactionView.test.js

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import TransactionView from '../../components/TransactionView';

describe('TransactionView component', () => {
    it('renders transaction list', () => {
        const transactions = [
            { id: 1, type: 'Deposit', amount: 100 },
            { id: 2, type: 'Withdrawal', amount: 50 },
            { id: 3, type: 'Transfer', amount: 200 },
        ];

        const { getAllByRole } = render(<TransactionView transactions={transactions} />);

        const transactionListItems = getAllByRole('listitem');

        expect(transactionListItems.length).toBe(transactions.length);

        transactions.forEach((transaction, index) => {
            expect(transactionListItems[index]).toHaveTextContent(`Transaction ${transaction.id}: ${transaction.type} $${transaction.amount}`);
        });
    });

    it('renders loading state when transactions are not provided', () => {
        const { getByText } = render(<TransactionView />);

        expect(getByText('Loading transactions...')).toBeInTheDocument();
    });

    it('handles transactions update', async () => {
        const transactions = [
            { id: 1, type: 'Deposit', amount: 100 },
            { id: 2, type: 'Withdrawal', amount: 50 },
            { id: 3, type: 'Transfer', amount: 200 },
        ];

        const updatedTransactions = [
            { id: 4, type: 'Deposit', amount: 300 },
            { id: 5, type: 'Withdrawal', amount: 100 },
            { id: 6, type: 'Transfer', amount: 400 },
        ];

        const { getAllByRole, rerender } = render(<TransactionView transactions={transactions} />);

        rerender(<TransactionView transactions={updatedTransactions} />);

        await waitFor(() => {
            const updatedTransactionListItems = getAllByRole('listitem');

            expect(updatedTransactionListItems.length).toBe(updatedTransactions.length);

            updatedTransactions.forEach((transaction, index) => {
                expect(updatedTransactionListItems[index]).toHaveTextContent(`Transaction ${transaction.id}: ${transaction.type } $${transaction.amount}`);
            });
        });
    });
});
