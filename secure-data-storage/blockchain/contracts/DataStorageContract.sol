// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DataStorageContract {
    // Struct to hold data entries
    struct DataEntry {
        bytes32 dataHash; // Hash of the encrypted data
        uint256 timestamp; // Timestamp of data storage
        address owner; // Owner of the data
    }

    // Mapping to store data entries by key
    mapping(string => DataEntry) private dataEntries;

    // Event emitted when data is stored
    event DataStored(string indexed key, bytes32 dataHash, address indexed owner, uint256 timestamp);

    // Event emitted when data is retrieved
    event DataRetrieved(string indexed key, address indexed owner, uint256 timestamp);

    // Modifier to check if the caller is the owner of the data
    modifier onlyOwner(string memory key) {
        require(msg.sender == dataEntries[key].owner, "Not the owner of the data");
        _;
    }

    // Function to store data
    function storeData(string memory key, bytes32 dataHash) public {
        require(dataEntries[key].owner == address(0), "Data already exists for this key");

        // Create a new data entry
        dataEntries[key] = DataEntry({
            dataHash: dataHash,
            timestamp: block.timestamp,
            owner: msg.sender
        });

        // Emit event for data storage
        emit DataStored(key, dataHash, msg.sender, block.timestamp);
    }

    // Function to retrieve data hash
    function retrieveData(string memory key) public view returns (bytes32) {
        require(dataEntries[key].owner != address(0), "Data does not exist");
        
        // Emit event for data retrieval
        emit DataRetrieved(key, dataEntries[key].owner, block.timestamp);
        
        return dataEntries[key].dataHash;
    }

    // Function to update data
    function updateData(string memory key, bytes32 newDataHash) public onlyOwner(key) {
        require(dataEntries[key].owner != address(0), "Data does not exist");

        // Update the data entry
        dataEntries[key].dataHash = newDataHash;

        // Emit event for data update
        emit DataStored(key, newDataHash, msg.sender, block.timestamp);
    }

    // Function to delete data
    function deleteData(string memory key) public onlyOwner(key) {
        require(dataEntries[key].owner != address(0), "Data does not exist");

        // Delete the data entry
        delete dataEntries[key];

        // Emit event for data deletion
        emit DataRetrieved(key, msg.sender, block.timestamp);
    }

    // Function to check if data exists
    function dataExists(string memory key) public view returns (bool) {
        return dataEntries[key].owner != address(0);
    }

    // Function to get the owner of the data
    function getDataOwner(string memory key) public view returns (address) {
        require(dataEntries[key].owner != address(0), "Data does not exist");
        return dataEntries[key].owner;
    }
}
