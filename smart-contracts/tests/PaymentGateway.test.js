// tests/PaymentGateway.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PaymentGateway", function () {
    let paymentGateway;
    let deliveryVerification;
    let owner;
    let buyer;
    let seller;

    beforeEach(async function () {
        [owner, buyer, seller] = await ethers.getSigners();

        const DeliveryVerification = await ethers.getContractFactory("DeliveryVerification");
        deliveryVerification = await DeliveryVerification.deploy();
        await deliveryVerification.deployed();

        const PaymentGateway = await ethers.getContractFactory("PaymentGateway");
        paymentGateway = await PaymentGateway.deploy(deliveryVerification.address);
        await paymentGateway.deployed();
    });

    it("should initiate a payment", async function () {
        const tradeAgreementId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await expect(paymentGateway.initiatePayment(tradeAgreementId, seller.address, { value: amount }))
            .to.emit(paymentGateway, "PaymentInitiated")
            .withArgs(1, tradeAgreementId, buyer.address, seller.address, amount);
    });

    it("should complete a payment after delivery confirmation", async function () {
        const tradeAgreementId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await paymentGateway.initiatePayment(tradeAgreementId, seller.address, { value: amount });
        await deliveryVerification.initiateDelivery(tradeAgreementId, buyer.address, "Delivery details");
        await deliveryVerification.confirmDelivery(1);

        await expect(paymentGateway.completePayment(tradeAgreementId))
            .to.emit(paymentGateway, "PaymentCompleted")
            .withArgs(tradeAgreementId, buyer.address, seller.address, amount);
    });

    it("should not complete payment if delivery is not confirmed", async function () {
        const tradeAgreementId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await paymentGateway.initiatePayment(tradeAgreementId, seller.address, { value: amount });
        await expect(paymentGateway.completePayment(tradeAgreementId))
            .to.be.revertedWith("PaymentGateway: Delivery must be confirmed before payment completion");
    });
});
