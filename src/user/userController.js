// src/user/userController.js
const UserService = require('./userService');
const { validationResult } = require('express-validator');
const logger = require('../utils/logger');

class UserController {
    // Register a new user
    async registerUser (req, res) {
        try {
            // Validate request input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({ errors: errors.array() });
            }

            const { username, email, password } = req.body;
            const newUser  = await UserService.registerUser (username, email, password);
            return res.status(201).json({ success: true, user: newUser  });
        } catch (error) {
            logger.error(`User  registration failed: ${error.message}`);
            return res.status(500).json({ success: false, message: error.message });
        }
    }

    // Authenticate a user
    async authenticateUser (req, res) {
        try {
            // Validate request input
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({ errors: errors.array() });
            }

            const { email, password } = req.body;
            const { user, token } = await UserService.authenticateUser (email, password);
            return res.status(200).json({ success: true, user, token });
        } catch (error) {
            logger.error(`User  authentication failed: ${error.message}`);
            return res.status(401).json({ success: false, message: error.message });
        }
    }

    // Get user by ID
    async getUser ById(req, res) {
        try {
            const { userId } = req.params;
            const user = await UserService.getUser ById(userId);
            return res.status(200).json({ success: true, user });
        } catch (error) {
            logger.error(`Failed to retrieve user: ${error.message}`);
            return res.status(404).json({ success: false, message: error.message });
        }
    }

    // Update user information
    async updateUser (req, res) {
        try {
            const { userId } = req.params;
            const updateData = req.body;
            const updatedUser  = await UserService.updateUser (userId, updateData);
            return res.status(200).json({ success: true, user: updatedUser  });
        } catch (error) {
            logger.error(`Failed to update user: ${error.message}`);
            return res.status(500).json({ success: false, message: error.message });
        }
    }
}

module.exports = new UserController();
