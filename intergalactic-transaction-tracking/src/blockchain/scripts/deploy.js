// blockchain/scripts/deploy.js

const { ethers } = require("hardhat");

async function main() {
    // Get the contract factory for TransactionTracker
    const TransactionTracker = await ethers.getContractFactory("TransactionTracker");

    // Deploy the contract
    const transactionTracker = await TransactionTracker.deploy();

    // Wait for the deployment to be mined
    await transactionTracker.deployed();

    console.log("TransactionTracker deployed to:", transactionTracker.address);
}

// Handle errors during deployment
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Error during deployment:", error);
        process.exit(1);
    });
