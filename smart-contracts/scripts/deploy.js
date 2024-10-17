// scripts/deploy.js
const { ethers } = require("hardhat");

async function main() {
    // Compile the contracts
    await hre.run('compile');

    // Get the contract factories
    const PaymentGateway = await ethers.getContractFactory("PaymentGateway");
    const DeliveryVerification = await ethers.getContractFactory("DeliveryVerification");

    // Deploy the contracts
    const paymentGateway = await PaymentGateway.deploy();
    await paymentGateway.deployed();
    console.log("PaymentGateway deployed to:", paymentGateway.address);

    const deliveryVerification = await DeliveryVerification.deploy();
    await deliveryVerification.deployed();
    console.log("DeliveryVerification deployed to:", deliveryVerification.address);
}

// Execute the deployment script
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
