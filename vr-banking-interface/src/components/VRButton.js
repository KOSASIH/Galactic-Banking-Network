// src/components/VRButton.js

import React from 'react';
import PropTypes from 'prop-types';
import { VR_MODE } from '../utils/constants';

const VRButton = ({ isVRMode, onToggleVR }) => {
    const handleClick = () => {
        onToggleVR(!isVRMode);
    };

    return (
        <button className={`vr-button ${isVRMode ? 'active' : ''}`} onClick={handleClick}>
            {isVRMode ? 'Exit VR Mode' : 'Enter VR Mode'}
        </button>
    );
};

VRButton.propTypes = {
    isVRMode: PropTypes.bool.isRequired,
    onToggleVR: PropTypes.func.isRequired,
};

export default VRButton;
