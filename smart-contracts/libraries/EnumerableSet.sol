// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

library EnumerableSet {
    struct AddressSet {
        address[] _values;
        mapping(address => uint256) _indexes; // value => index
    }

    function add(AddressSet storage set, address value) internal returns (bool) {
        if (!contains(set, value)) {
            set._values.push(value);
            set._indexes[value] = set._values.length; // Store the index + 1
            return true;
        }
        return false;
    }

    function remove(AddressSet storage set, address value) internal returns (bool) {
        uint256 valueIndex = set._indexes[value];

        if (valueIndex != 0) { // Equivalent to contains(set, value)
            uint256 toDeleteIndex = valueIndex - 1;
            uint256 lastIndex = set._values.length - 1;

            if (lastIndex != toDeleteIndex) {
                address lastValue = set._values[lastIndex];
                set._values[toDeleteIndex] = lastValue;
                set._indexes[lastValue] = valueIndex; // Update the index
            }

            set._values.pop();
            delete set._indexes[value];

            return true;
        }
        return false;
    }

    function contains(AddressSet storage set, address value) internal view returns (bool) {
        return set._indexes[value] != 0;
    }

    function length(AddressSet storage set) internal view returns (uint256) {
        return set._values.length;
    }

    function at(AddressSet storage set, uint256 index) internal view returns (address) {
        require(index < length(set), "EnumerableSet: index out of bounds");
        return set._values[index];
    }

    function values(AddressSet storage set) internal view returns (address[] memory) {
        return set._values;
    }
}
