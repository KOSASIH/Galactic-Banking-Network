// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IDeliveryVerification {
    enum DeliveryStatus { Pending, Delivered, Disputed }

    struct Delivery {
        uint256 tradeAgreementId;
        address seller;
        address buyer;
        DeliveryStatus status;
        uint256 deliveredAt;
        string deliveryDetails; // Details about delivery
        string disputeReason; // Reason for dispute, if any
    }

    event DeliveryInitiated(uint256 indexed deliveryId, uint256 indexed tradeAgreementId, address indexed seller, address buyer, string deliveryDetails);
    event DeliveryConfirmed(uint256 indexed deliveryId, uint256 deliveredAt);
    event DeliveryDisputed(uint256 indexed deliveryId, string reason);

    function initiateDelivery(uint256 tradeAgreementId, address buyer, string memory deliveryDetails) external;

    function confirmDelivery(uint256 deliveryId) external;

    function disputeDelivery(uint256 deliveryId, string memory reason) external;

    function getDelivery(uint256 deliveryId) external view returns (Delivery memory);
}
