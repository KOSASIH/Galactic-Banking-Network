// src/user/userService.js
const User = require('./userModel');
const jwt = require('jsonwebtoken');
const logger = require('../utils/logger');

class UserService {
    constructor() {
        this.secretKey = process.env.JWT_SECRET || 'your_jwt_secret'; // Use environment variable for security
    }

    // Register a new user
    async registerUser (username, email, password) {
        try {
            const existingUser  = await User.findOne({ email });
            if (existingUser ) {
                throw new Error('Email already in use');
            }

            const newUser  = new User({ username, email, password });
            await newUser .save();
            logger.info(`User  registered successfully: ${newUser .username}`);
            return newUser ;
        } catch (error) {
            logger.error(`User  registration failed: ${error.message}`);
            throw new Error(`Registration failed: ${error.message}`);
        }
    }

    // Authenticate a user
    async authenticateUser (email, password) {
        try {
            const user = await User.findOne({ email });
            if (!user) {
                throw new Error('Invalid email or password');
            }

            const isMatch = await user.comparePassword(password);
            if (!isMatch) {
                throw new Error('Invalid email or password');
            }

            // Generate JWT token
            const token = this.generateToken(user._id);
            logger.info(`User  authenticated successfully: ${user.username}`);
            return { user, token };
        } catch (error) {
            logger.error(`User  authentication failed: ${error.message}`);
            throw new Error(`Authentication failed: ${error.message}`);
        }
    }

    // Generate JWT token
    generateToken(userId) {
        return jwt.sign({ id: userId }, this.secretKey, { expiresIn: '1h' });
    }

    // Get user by ID
    async getUser ById(userId) {
        try {
            const user = await User.findById(userId);
            if (!user) {
                throw new Error('User  not found');
            }
            logger.info(`Retrieved user: ${user.username}`);
            return user;
        } catch (error) {
            logger.error(`Failed to retrieve user: ${error.message}`);
            throw new Error(`Could not retrieve user: ${error.message}`);
        }
    }

    // Update user information
    async updateUser (userId, updateData) {
        try {
            const user = await User.findByIdAndUpdate(userId, updateData, { new: true });
            if (!user) {
                throw new Error('User  not found');
            }
            logger.info(`User  updated successfully: ${user.username}`);
            return user;
        } catch (error) {
            logger.error(`Failed to update user: ${error.message}`);
            throw new Error(`Could not update user: ${error.message}`);
        }
    }
}

module.exports = new UserService();
