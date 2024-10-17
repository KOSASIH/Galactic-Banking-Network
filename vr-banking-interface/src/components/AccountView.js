// src/components/AccountView.js

import React, { useEffect, useState } from 'react';
import AccountService from '../services/AccountService';
import { VR_MODE } from '../utils/constants';

const AccountView = ({ userId, isVRMode }) => {
    const [account, setAccount] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchAccountDetails = async () => {
            try {
                setLoading(true);
                const accountDetails = await AccountService.getAccountDetails(userId);
                setAccount(accountDetails);
            } catch (err) {
                setError('Failed to load account details.');
            } finally {
                setLoading(false);
            }
        };

        fetchAccountDetails();
    }, [userId]);

    if (loading) {
        return <div className="loading">Loading account details...</div>;
    }

    if (error) {
        return <div className="error">{error}</div>;
    }

    return (
        <div className={`account-view ${isVRMode ? 'vr-mode' : ''}`}>
            <h1>Account Balance: {account.balance} GCR</h1>
            <h2>Account Holder: {account.holderName}</h2>
            <div className="account-actions">
                <button className="action-button" onClick={() => alert('Deposit functionality coming soon!')}>
                    Deposit
                </button>
                <button className="action-button" onClick={() => alert('Withdraw functionality coming soon!')}>
                    Withdraw
                </button>
            </div>
            <style jsx>{`
                .account-view {
                    padding: 20px;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    transition: transform 0.3s;
                }
                .vr-mode {
                    transform: scale(1.1);
                    background-color: rgba(255, 255, 255, 0.9);
                }
                .loading, .error {
                    font-size: 1.2em;
                    color: #333;
                }
                .account-actions {
                    margin-top: 20px;
                }
                .action-button {
                    margin: 5px;
                    padding: 10px 20px;
                    font-size: 1em;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    background-color: #007bff;
                    color: white;
                    transition: background-color 0.3s;
                }
                .action-button:hover {
                    background-color: #0056b3;
                }
            `}</style>
        </div>
    );
};

export default AccountView;
