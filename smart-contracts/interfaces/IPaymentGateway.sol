// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IPaymentGateway {
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

    event PaymentInitiated(uint256 indexed paymentId, uint256 indexed tradeAgreementId, address indexed buyer, address seller, uint256 amount);
    event PaymentCompleted(uint256 indexed paymentId, uint256 completedAt);
    event PaymentRefunded(uint256 indexed paymentId, uint256 refundedAt);

    function initiatePayment(uint256 tradeAgreementId, address seller) external payable;

    function completePayment(uint256 paymentId) external;

    function refundPayment(uint256 paymentId) external;

    function getPayment(uint256 paymentId) external view returns (Payment memory);
}
