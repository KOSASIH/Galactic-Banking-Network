// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./interfaces/ITradeAgreement.sol";
import "./libraries/SafeMath.sol";

contract TradeAgreement is ITradeAgreement {
    using SafeMath for uint256;

    enum AgreementStatus { Created, Fulfilled, Disputed, Canceled }

    struct TradeAgreementDetails {
        address buyer;
        address seller;
        uint256 price;
        uint256 quantity;
        AgreementStatus status;
        uint256 createdAt;
        uint256 fulfilledAt;
        string deliveryDetails; // Details about delivery
    }

    mapping (uint256 => TradeAgreementDetails) public tradeAgreements;
    uint256 public agreementCount;

    event TradeAgreementCreated(uint256 indexed tradeAgreementId, address indexed buyer, address indexed seller, uint256 price, uint256 quantity);
    event TradeAgreementFulfilled(uint256 indexed tradeAgreementId, uint256 fulfilledAt);
    event TradeAgreementDisputed(uint256 indexed tradeAgreementId, string reason);
    event TradeAgreementCanceled(uint256 indexed tradeAgreementId);

    modifier onlyBuyer(uint256 tradeAgreementId) {
        require(msg.sender == tradeAgreements[tradeAgreementId].buyer, "Only buyer can call this function");
        _;
    }

    modifier onlySeller(uint256 tradeAgreementId) {
        require(msg.sender == tradeAgreements[tradeAgreementId].seller, "Only seller can call this function");
        _;
    }

    modifier inStatus(uint256 tradeAgreementId, AgreementStatus status) {
        require(tradeAgreements[tradeAgreementId].status == status, "Invalid agreement status");
        _;
    }

    // Function to create a new trade agreement
    function createTradeAgreement(address seller, uint256 price, uint256 quantity, string memory deliveryDetails) public {
        require(seller != address(0), "Invalid seller address");
        require(price > 0, "Price must be greater than zero");
        require(quantity > 0, "Quantity must be greater than zero");

        agreementCount++;
        uint256 tradeAgreementId = agreementCount;

        TradeAgreementDetails memory tradeAgreementDetails = TradeAgreementDetails({
            buyer: msg.sender,
            seller: seller,
            price: price,
            quantity: quantity,
            status: AgreementStatus.Created,
            createdAt: block.timestamp,
            fulfilledAt: 0,
            deliveryDetails: deliveryDetails
        });

        tradeAgreements[tradeAgreementId] = tradeAgreementDetails;

        emit TradeAgreementCreated(tradeAgreementId, msg.sender, seller, price, quantity);
    }

    // Function to fulfill a trade agreement
    function fulfillTradeAgreement(uint256 tradeAgreementId) public onlySeller(tradeAgreementId) inStatus(tradeAgreementId, AgreementStatus.Created) {
        tradeAgreements[tradeAgreementId].status = AgreementStatus.Fulfilled;
        tradeAgreements[tradeAgreementId].fulfilledAt = block.timestamp;

        emit TradeAgreementFulfilled(tradeAgreementId, block.timestamp);
    }

    // Function to dispute a trade agreement
    function disputeTradeAgreement(uint256 tradeAgreementId, string memory reason) public onlyBuyer(tradeAgreementId) inStatus(tradeAgreementId, AgreementStatus.Created) {
        tradeAgreements[tradeAgreementId].status = AgreementStatus.Disputed;

        emit TradeAgreementDisputed(tradeAgreementId, reason);
    }

    // Function to cancel a trade agreement
    function cancelTradeAgreement(uint256 tradeAgreementId) public {
        require(msg.sender == tradeAgreements[tradeAgreementId].buyer || msg.sender == tradeAgreements[tradeAgreementId].seller, "Only buyer or seller can cancel");
        require(tradeAgreements[tradeAgreementId].status == AgreementStatus.Created, "Agreement cannot be canceled");

        tradeAgreements[tradeAgreementId].status = AgreementStatus.Canceled;

        emit TradeAgreementCanceled(tradeAgreementId);
    }

    // Function to get trade agreement details
    function getTradeAgreement(uint256 tradeAgreementId) public view returns (TradeAgreementDetails memory) {
        return tradeAgreements[tradeAgreementId];
    }
}
