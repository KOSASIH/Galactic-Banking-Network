// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./interfaces/IDeliveryVerification.sol";
import "./libraries/SafeMath.sol";

contract DeliveryVerification is IDeliveryVerification {
    using SafeMath for uint256;

    enum DeliveryStatus { Pending, Delivered, Disputed }

    struct Delivery {
        uint256 tradeAgreementId;
        address seller;
        address buyer;
        DeliveryStatus status;
        uint256 deliveredAt;
        string deliveryDetails; // Details about the delivery
        string disputeReason; // Reason for dispute, if any
    }

    mapping(uint256 => Delivery) public deliveries;
    uint256 public deliveryCount;

    event DeliveryInitiated(uint256 indexed deliveryId, uint256 indexed tradeAgreementId, address indexed seller, address buyer, string deliveryDetails);
    event DeliveryConfirmed(uint256 indexed deliveryId, uint256 deliveredAt);
    event DeliveryDisputed(uint256 indexed deliveryId, string reason);

    modifier onlySeller(uint256 deliveryId) {
        require(msg.sender == deliveries[deliveryId].seller, "Only seller can call this function");
        _;
    }

    modifier onlyBuyer(uint256 deliveryId) {
        require(msg.sender == deliveries[deliveryId].buyer, "Only buyer can call this function");
        _;
    }

    modifier inStatus(uint256 deliveryId, DeliveryStatus status) {
        require(deliveries[deliveryId].status == status, "Invalid delivery status");
        _;
    }

    // Function to initiate a delivery
    function initiateDelivery(uint256 tradeAgreementId, address buyer, string memory deliveryDetails) public {
        require(buyer != address(0), "Invalid buyer address");

        deliveryCount++;
        uint256 deliveryId = deliveryCount;

        Delivery memory delivery = Delivery({
            tradeAgreementId: tradeAgreementId,
            seller: msg.sender,
            buyer: buyer,
            status: DeliveryStatus.Pending,
            deliveredAt: 0,
            deliveryDetails: deliveryDetails,
            disputeReason: ""
        });

        deliveries[deliveryId] = delivery;

        emit DeliveryInitiated(deliveryId, tradeAgreementId, msg.sender, buyer, deliveryDetails);
    }

    // Function to confirm delivery
    function confirmDelivery(uint256 deliveryId) public onlySeller(deliveryId) inStatus(deliveryId, DeliveryStatus.Pending) {
        Delivery storage delivery = deliveries[deliveryId];
        delivery.status = DeliveryStatus.Delivered;
        delivery.deliveredAt = block.timestamp;

        emit DeliveryConfirmed(deliveryId, block.timestamp);
    }

    // Function to dispute a delivery
    function disputeDelivery(uint256 deliveryId, string memory reason) public onlyBuyer(deliveryId) inStatus(deliveryId, DeliveryStatus.Delivered) {
        Delivery storage delivery = deliveries[deliveryId];
        delivery.status = DeliveryStatus.Disputed;
        delivery.disputeReason = reason;

        emit DeliveryDisputed(deliveryId, reason);
    }

    // Function to get delivery details
    function getDelivery(uint256 deliveryId) public view returns (Delivery memory) {
        return deliveries[deliveryId];
    }
}
