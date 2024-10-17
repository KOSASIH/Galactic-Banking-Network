// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface ITradeAgreement {
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

    event TradeAgreementCreated(uint256 indexed tradeAgreementId, address indexed buyer, address indexed seller, uint256 price, uint256 quantity);
    event TradeAgreementFulfilled(uint256 indexed tradeAgreementId, uint256 fulfilledAt);
    event TradeAgreementDisputed(uint256 indexed tradeAgreementId, string reason);
    event TradeAgreementCanceled(uint256 indexed tradeAgreementId);

    function createTradeAgreement(address seller, uint256 price, uint256 quantity, string memory deliveryDetails) external;

    function fulfillTradeAgreement(uint256 tradeAgreementId) external;

    function disputeTradeAgreement(uint256 tradeAgreementId, string memory reason) external;

    function cancelTradeAgreement(uint256 tradeAgreementId) external;

    function getTradeAgreement(uint256 tradeAgreementId) external view returns (TradeAgreementDetails memory);
}
