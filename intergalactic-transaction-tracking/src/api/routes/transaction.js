// src/api/routes/transaction.js

const express = require('express');
const router = express.Router();
const TransactionController = require('../controllers/TransactionController');
const { body, param, validationResult } = require('express-validator');

// Middleware for validating transaction data
const validateTransactionData = [
  body('sender').isString().withMessage('Sender must be a string'),
  body('recipient').isString().withMessage('Recipient must be a string'),
  body('amount').isNumeric().withMessage('Amount must be a number').isFloat({ gt: 0 }).withMessage('Amount must be greater than zero'),
  body('currency').isString().withMessage('Currency must be a string'),
  body('planet').isString().withMessage('Planet must be a string'),
];

// Route to get a transaction by ID
router.get('/:id', 
  param('id').isMongoId().withMessage('Invalid transaction ID'), 
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    await TransactionController.getTransaction(req, res);
});

// Route to create a new transaction
router.post('/', validateTransactionData, async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  await TransactionController.createTransaction(req, res);
});

// Route to update a transaction status
router.patch('/:id/status', 
  param('id').isMongoId().withMessage('Invalid transaction ID'),
  body('status').isIn(['pending', 'completed', 'failed']).withMessage('Status must be one of: pending, completed, failed'),
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    await TransactionController.updateTransactionStatus(req, res);
});

// Route to get all transactions for a specific sender
router.get('/sender/:senderId', 
  param('senderId').isString().withMessage('Invalid sender ID'), 
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }
    await TransactionController.getTransactionsBySender(req, res);
});

module.exports = router;
