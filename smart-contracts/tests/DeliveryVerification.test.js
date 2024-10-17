// tests/DeliveryVerification.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("DeliveryVerification", function () {
    let deliveryVerification;
    let owner;
    let buyer;
    let seller;

    beforeEach(async function () {
        [owner, buyer, seller] = await ethers.getSigners();

        const DeliveryVerification = await ethers.getContractFactory("DeliveryVerification");
        deliveryVerification = await DeliveryVerification.deploy();
        await deliveryVerification.deployed();
    });

    it("should initiate delivery", async function () {
        const tradeAgreementId = 1;
        const deliveryDetails = "Package delivered to the front door.";

        await expect(deliveryVerification.initiateDelivery(tradeAgreementId, buyer.address, deliveryDetails))
            .to.emit(deliveryVerification, "DeliveryInitiated")
            .withArgs(tradeAgreementId, buyer.address, deliveryDetails);
    });

    it("should confirm delivery", async function () {
        const tradeAgreementId = 1;
        const deliveryDetails = "Package delivered to the front door.";

        await deliveryVerification.initiateDelivery(tradeAgreementId, buyer.address, deliveryDetails);
        await expect(deliveryVerification.confirmDelivery(tradeAgreementId))
            .to.emit(deliveryVerification, "DeliveryConfirmed")
            .withArgs(tradeAgreementId, (await ethers.provider.getBlock('latest')).timestamp);
    });

    it("should dispute delivery", async function () {
        const tradeAgreementId = 1;
        const deliveryDetails = "Package delivered to the front door.";
        const disputeReason = "Package was damaged.";

        await deliveryVerification.initiateDelivery(tradeAgreementId, buyer.address, deliveryDetails);
        await expect(deliveryVerification.disputeDelivery(tradeAgreementId, disputeReason))
            .to.emit(deliveryVerification, "DeliveryDisputed")
            .withArgs(tradeAgreementId, disputeReason);
    });
});
