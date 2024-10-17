// src/components/TransactionView.js

import React, { useEffect, useState } from 'react';
import TransactionService from '../services/TransactionService';
import { VR_MODE } from '../utils/constants';

const TransactionView = ({ userId, isVRMode }) => {
    const [transactions, setTransactions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchTransactions = async () => {
            try {
                setLoading(true);
                const transactionList = await TransactionService.getRecentTransactions(userId);
                setTransactions(transactionList);
            } catch (err) {
                setError('Failed to load transactions.');
            } finally {
                setLoading(false);
            }
        };

        fetchTransactions();
    }, [userId]);

    if (loading) {
        return <div className="loading">Loading transactions...</div>;
    }

    if (error) {
        return <div className="error">{error}</div>;
    }

    return (
        <div className={`transaction-view ${isVRMode ? 'vr-mode' : ''}`}>
            <h2>Recent Transactions</h2>
            <ul className="transaction-list">
                {transactions.map((transaction) => (
                    <li key={transaction.id} className="transaction-item">
                        <span className="transaction-date">{transaction.date}</span>
                        <span className="transaction-amount">{transaction.amount} GCR</span>
                        <span className="transaction-type">{transaction.type}</span>
                    </li>
                ))}
            </ul>
            <style jsx>{`
                .transaction-view {
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
                .transaction-list {
                    list-style-type: none;
                    padding: 0;
                }
                .transaction-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }
                .transaction-date {
                    flex: 1;
                }
                .transaction-amount {
                    flex: 1;
                    font-weight: bold;
                }
                .transaction-type {
                    flex: 1;
                    color: #888;
                }
            `}</style>
        </div>
    );
};

export default TransactionView;
