// tests/TradeAgreement.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("TradeAgreement", function () {
    let tradeAgreement;
    let owner;
    let buyer;
    let seller;

    beforeEach(async function () {
        [owner, buyer, seller] = await ethers.getSigners();

        const TradeAgreement = await ethers.getContractFactory("TradeAgreement");
        tradeAgreement = await TradeAgreement.deploy();
        await tradeAgreement.deployed();
    });

    it("should create a trade agreement", async function () {
        const tradeId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await expect(tradeAgreement.createAgreement(tradeId, seller.address, buyer.address, amount))
            .to.emit(tradeAgreement, "AgreementCreated")
            .withArgs(tradeId, seller.address, buyer.address, amount);
    });

    it("should retrieve a trade agreement", async function () {
        const tradeId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await tradeAgreement.createAgreement(tradeId, seller.address, buyer.address, amount);
        const agreement = await tradeAgreement.getAgreement(tradeId);

        expect(agreement.seller).to.equal(seller.address);
        expect(agreement.buyer).to.equal(buyer.address);
        expect(agreement.amount.toString()).to.equal(amount.toString());
    });

    it("should not allow duplicate trade agreements", async function () {
        const tradeId = 1;
        const amount = ethers.utils.parseEther("1.0");

        await tradeAgreement.createAgreement(tradeId, seller.address, buyer.address, amount);
        await expect(tradeAgreement.createAgreement(tradeId, seller.address, buyer.address, amount))
            .to.be.revertedWith("TradeAgreement: Agreement already exists");
    });
});
