// src/App.js

import React, { useState, useEffect } from 'react';
import VRButton from './components/VRButton';
import AccountView from './components/AccountView';
import TransactionView from './components/TransactionView';
import { VR_MODE } from './utils/constants';
import VRService from './services/VRService';
import AccountService from './services/AccountService';

const App = () => {
    const [isVRMode, setIsVRMode] = useState(false);
    const [accountDetails, setAccountDetails] = useState(null);
    const [transactions, setTransactions] = useState([]);

    useEffect(() => {
        const fetchAccountDetails = async () => {
            const accountData = await AccountService.getAccountDetails('user123');
            setAccountDetails(accountData);
        };

        const fetchTransactions = async () => {
            // Simulate fetching transactions from an API
            const transactionData = [
                { id: 1, type: 'Deposit', amount: 100 },
                { id: 2, type: 'Withdrawal', amount: 50 },
                { id: 3, type: 'Transfer', amount: 200 },
            ];
            setTransactions(transactionData);
        };

        fetchAccountDetails();
        fetchTransactions();
    }, []);

    const handleVRModeChange = async (mode) => {
        if (mode) {
            await VRService.enterVR();
        } else {
            await VRService.exitVR();
        }
        setIsVRMode(mode);
    };

    return (
        <div className="app">
            <VRButton isVRMode={isVRMode} onToggleVR={handleVRModeChange} />
            {accountDetails && (
                <AccountView
                    balance={accountDetails.balance}
                    holderName={accountDetails.holderName}
                    accountId={accountDetails.accountId}
                />
            )}
            {transactions.length > 0 && (
                <TransactionView transactions={transactions} />
            )}
        </div>
    );
};

export default App;
