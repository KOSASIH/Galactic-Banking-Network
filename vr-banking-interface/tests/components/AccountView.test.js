// tests/components/AccountView.test.js

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import AccountView from '../../components/AccountView';

describe('AccountView component', () => {
    it('renders account details', () => {
        const accountDetails = {
            balance: 1000,
            holderName: 'John Doe',
            accountId: '1234567890',
        };

        const { getByText } = render(<AccountView {...accountDetails} />);

        expect(getByText(`Account Balance: $${accountDetails.balance}`)).toBeInTheDocument();
        expect(getByText(`Account Holder: ${accountDetails.holderName}`)).toBeInTheDocument();
        expect(getByText(`Account ID: ${accountDetails.accountId}`)).toBeInTheDocument();
    });

    it('renders loading state when account details are not provided', () => {
        const { getByText } = render(<AccountView />);

        expect(getByText('Loading account details...')).toBeInTheDocument();
    });

    it('handles account details update', async () => {
        const accountDetails = {
            balance: 1000,
            holderName: 'John Doe',
            accountId: '1234567890',
        };

        const updatedAccountDetails = {
            balance: 2000,
            holderName: 'Jane Doe',
            accountId: '9876543210',
        };

        const { getByText, rerender } = render(<AccountView {...accountDetails} />);

        rerender(<AccountView {...updatedAccountDetails} />);

        await waitFor(() => {
            expect(getByText(`Account Balance: $${updatedAccountDetails.balance}`)).toBeInTheDocument();
            expect(getByText(`Account Holder: ${updatedAccountDetails.holderName}`)).toBeInTheDocument();
            expect(getByText(`Account ID: ${updatedAccountDetails.accountId}`)).toBeInTheDocument();
        });
    });
});
