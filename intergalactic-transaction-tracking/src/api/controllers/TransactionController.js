// src/api/controllers/TransactionController.js

const express = require('express');
const router = express.Router();
const TransactionService = require('../services/TransactionService');

// Get transaction by ID
router.get('/:id', async (req, res) => {
  try {
    const transactionId = req.params.id;
    const transaction = await TransactionService.getTransaction(transactionId);
    
    if (!transaction) {
      return res.status(404).json({ message: 'Transaction not found' });
    }
    
    res.json(transaction);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Create a new transaction
router.post('/', async (req, res) => {
  try {
    const transactionData = req.body;
    
    // Validate transaction data
    if (!transactionData.sender || !transactionData.recipient || !transactionData.amount || !transactionData.currency) {
      return res.status(400).json({ message: 'Invalid transaction data' });
    }
    
    const transaction = await TransactionService.createTransaction(transactionData);
    res.status(201).json(transaction);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;
