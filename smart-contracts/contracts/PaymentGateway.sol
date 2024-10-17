// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./interfaces/IPaymentGateway.sol";
import "./libraries/SafeMath.sol";

contract PaymentGateway is IPaymentGateway {
    using SafeMath for uint256;

    enum PaymentStatus { Pending, Completed, Refunded }

    struct Payment {
        uint256 tradeAgreementId;
        address buyer;
        address seller;
        uint256 amount;
        PaymentStatus status;
        uint256 createdAt;
        uint256 completedAt;
    }

    mapping(uint256 => Payment) public payments;
    uint256 public paymentCount;

    event PaymentInitiated(uint256 indexed paymentId, uint256 indexed tradeAgreementId, address indexed buyer, address seller, uint256 amount);
    event PaymentCompleted(uint256 indexed paymentId, uint256 completedAt);
    event PaymentRefunded(uint256 indexed paymentId, uint256 refundedAt);

    modifier onlyBuyer(uint256 paymentId) {
        require(msg.sender == payments[paymentId].buyer, "Only buyer can call this function");
        _;
    }

    modifier onlySeller(uint256 paymentId) {
        require(msg.sender == payments[paymentId].seller, "Only seller can call this function");
        _;
    }

    modifier inStatus(uint256 paymentId, PaymentStatus status) {
        require(payments[paymentId].status == status, "Invalid payment status");
        _;
    }

    // Function to initiate a payment
    function initiatePayment(uint256 tradeAgreementId, address seller) public payable {
        require(seller != address(0), "Invalid seller address");
        require(msg.value > 0, "Payment amount must be greater than zero");

        paymentCount++;
        uint256 paymentId = paymentCount;

        Payment memory payment = Payment({
            tradeAgreementId: tradeAgreementId,
            buyer: msg.sender,
            seller: seller,
            amount: msg.value,
            status: PaymentStatus.Pending,
            createdAt: block.timestamp,
            completedAt: 0
        });

        payments[paymentId] = payment;

        emit PaymentInitiated(paymentId, tradeAgreementId, msg.sender, seller, msg.value);
    }

    // Function to complete a payment
    function completePayment(uint256 paymentId) public onlySeller(paymentId) inStatus(paymentId, PaymentStatus.Pending) {
        Payment storage payment = payments[paymentId];
        payment.status = PaymentStatus.Completed;
        payment.completedAt = block.timestamp;

        // Transfer funds to the seller
        payable(payment.seller).transfer(payment.amount);

        emit PaymentCompleted(paymentId, block.timestamp);
    }

    // Function to refund a payment
    function refundPayment(uint256 paymentId) public onlySeller(paymentId) inStatus(paymentId, PaymentStatus.Pending) {
        Payment storage payment = payments[paymentId];
        payment.status = PaymentStatus.Refunded;

        // Refund the buyer
        payable(payment.buyer).transfer(payment.amount);

        emit PaymentRefunded(paymentId, block.timestamp);
    }

    // Function to get payment details
    function getPayment(uint256 paymentId) public view returns (Payment memory) {
        return payments[paymentId];
    }
}
