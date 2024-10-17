// scripts/test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PaymentGateway and DeliveryVerification", function () {
    let paymentGateway;
    let deliveryVerification;
    let owner;
    let buyer;
    let seller;

    beforeEach(async function () {
        [owner, buyer, seller] = await ethers.getSigners();

        const PaymentGateway = await ethers.getContractFactory("PaymentGateway");
        paymentGateway = await PaymentGateway.deploy();
        await paymentGateway.deployed();

        const DeliveryVerification = await ethers.getContractFactory("DeliveryVerification");
        deliveryVerification = await DeliveryVerification.deploy();
        await deliveryVerification.deployed();
    });

    it("should initiate a payment", async function () {
        const tradeAgreementId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await expect(paymentGateway.initiatePayment(tradeAgreementId, seller.address, { value: amount }))
            .to.emit(paymentGateway, "PaymentInitiated")
            .withArgs(1, tradeAgreementId, buyer.address, seller.address, amount);
    });

    it("should confirm delivery", async function () {
        const tradeAgreementId = 1;
        const deliveryDetails = "Package delivered to the front door.";

        await deliveryVerification.initiateDelivery(tradeAgreementId, buyer.address, deliveryDetails);
        await expect(deliveryVerification.confirmDelivery(1))
            .to.emit(deliveryVerification, "DeliveryConfirmed")
            .withArgs(1, (await ethers.provider.getBlock('latest')).timestamp);
    });

    it("should dispute a delivery", async function () {
        const tradeAgreementId = 1;
        const deliveryDetails = "Package delivered to the front door.";
        const disputeReason = "Package was damaged.";

        await deliveryVerification.initiateDelivery(tradeAgreementId, buyer.address, deliveryDetails);
        await expect(deliveryVerification.disputeDelivery(1, disputeReason))
            .to.emit(deliveryVerification, "DeliveryDisputed")
            .withArgs(1, disputeReason);
    });
});
