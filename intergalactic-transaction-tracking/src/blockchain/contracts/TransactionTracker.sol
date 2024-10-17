// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/AccessControl.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Counters.sol";

contract TransactionTracker is AccessControl {
    // Define the roles for the contract
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN");
    bytes32 public constant USER_ROLE = keccak256("USER");

    // Define the transaction structure
    struct Transaction {
        uint256 id;
        address sender;
        address recipient;
        uint256 amount;
        string currency;
        string planet;
        string status;
    }

    // Define the transaction mapping
    mapping (uint256 => Transaction) public transactions;

    // Define the transaction counter
    using Counters for Counters.Counter;
    Counters.Counter public transactionCounter;

    // Define the event for transaction creation
    event TransactionCreated(uint256 id, address sender, address recipient, uint256 amount, string currency, string planet);

    // Define the event for transaction update
    event TransactionUpdated(uint256 id, string status);

    // Constructor to initialize the contract
    constructor() {
        // Initialize the admin role
        _setupRole(ADMIN_ROLE, msg.sender);

        // Initialize the user role
        _setupRole(USER_ROLE, msg.sender);
    }

    // Function to create a new transaction
    function createTransaction(address sender, address recipient, uint256 amount, string memory currency, string memory planet) public onlyRole(USER_ROLE) {
        // Increment the transaction counter
        transactionCounter.increment();

        // Get the transaction ID
        uint256 transactionId = transactionCounter.current();

        // Create a new transaction
        transactions[transactionId] = Transaction(transactionId, sender, recipient, amount, currency, planet, "pending");

        // Emit the transaction creation event
        emit TransactionCreated(transactionId, sender, recipient, amount, currency, planet);
    }

    // Function to update a transaction status
    function updateTransactionStatus(uint256 transactionId, string memory status) public onlyRole(USER_ROLE) {
        // Check if the transaction exists
        require(transactions[transactionId].id != 0, "Transaction not found");

        // Update the transaction status
        transactions[transactionId].status = status;

        // Emit the transaction update event
        emit TransactionUpdated(transactionId, status);
    }

    // Function to get a transaction by ID
    function getTransaction(uint256 transactionId) public view returns (Transaction memory) {
        // Check if the transaction exists
        require(transactions[transactionId].id != 0, "Transaction not found");

        // Return the transaction
        return transactions[transactionId];
    }
}
