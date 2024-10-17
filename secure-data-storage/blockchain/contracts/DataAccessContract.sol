// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./DataStorageContract.sol"; // Importing the DataStorageContract

contract DataAccessContract {
    // Enum for access levels
    enum AccessLevel { NONE, READ, WRITE, ADMIN }

    // Struct to hold user access permissions
    struct UserAccess {
        AccessLevel accessLevel; // Access level of the user
        uint256 timestamp; // Timestamp of the last update
    }

    // Mapping to store user access permissions for each data key
    mapping(string => mapping(address => UserAccess)) private userAccess;

    // Reference to the DataStorageContract
    DataStorageContract private dataStorage;

    // Event emitted when access is granted
    event AccessGranted(string indexed key, address indexed user, AccessLevel accessLevel, uint256 timestamp);

    // Event emitted when access is revoked
    event AccessRevoked(string indexed key, address indexed user, uint256 timestamp);

    // Constructor to set the DataStorageContract address
    constructor(address _dataStorageAddress) {
        dataStorage = DataStorageContract(_dataStorageAddress);
    }

    // Modifier to check if the user has at least READ access
    modifier hasReadAccess(string memory key) {
        require(userAccess[key][msg.sender].accessLevel >= AccessLevel.READ, "No read access");
        _;
    }

    // Modifier to check if the user has WRITE access
    modifier hasWriteAccess(string memory key) {
        require(userAccess[key][msg.sender].accessLevel >= AccessLevel.WRITE, "No write access");
        _;
    }

    // Function to grant access to a user
    function grantAccess(string memory key, address user, AccessLevel accessLevel) public {
        require(dataStorage.dataExists(key), "Data does not exist");
        require(accessLevel > AccessLevel.NONE, "Invalid access level");

        // Grant access to the user
        userAccess[key][user] = UserAccess({
            accessLevel: accessLevel,
            timestamp: block.timestamp
        });

        // Emit event for access granted
        emit AccessGranted(key, user, accessLevel, block.timestamp);
    }

    // Function to revoke access from a user
    function revokeAccess(string memory key, address user) public {
        require(dataStorage.dataExists(key), "Data does not exist");
        require(userAccess[key][user].accessLevel > AccessLevel.NONE, "No access to revoke");

        // Revoke access
        delete userAccess[key][user];

        // Emit event for access revoked
        emit AccessRevoked(key, user, block.timestamp);
    }

    // Function to check user access level
    function checkAccess(string memory key, address user) public view returns (AccessLevel) {
        return userAccess[key][user].accessLevel;
    }

    // Function to read data if the user has access
    function readData(string memory key) public view hasReadAccess(key) returns (bytes32) {
        return dataStorage.retrieveData(key);
    }

    // Function to write data if the user has access
    function writeData(string memory key, bytes32 newDataHash) public hasWriteAccess(key) {
        dataStorage.updateData(key, newDataHash);
    }

    // Function to get the last access timestamp for a user
    function getLastAccessTimestamp(string memory key, address user) public view returns (uint256) {
        return userAccess[key][user].timestamp;
    }
}
