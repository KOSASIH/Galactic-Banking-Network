// Import required modules
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const logger = require('../utils/logger');
const errorHandler = require('../utils/errorHandler');
const notificationService = require('../utils/notificationService');

// Define the user schema
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: { type: String, required: true, enum: ['admin', 'user'] },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// Create the user model
const User = mongoose.model('User ', userSchema);

// Define the user service
const userService = {
  // Register a new user
  register: async (username, email, password) => {
    try {
      // Hash the password
      const hashedPassword = await bcrypt.hash(password, 10);

      // Create a new user document
      const user = new User({
        username,
        email,
        password: hashedPassword,
        role: 'user'
      });

      // Save the user to the database
      await user.save();

      // Send a notification to the user
      notificationService.sendNotification(`Welcome to our platform, ${username}!`);

      // Return the user document
      return user;
    } catch (error) {
      // Throw the error
      throw error;
    }
  },

  // Login a user
  login: async (username, password) => {
    try {
      // Find the user by username
      const user = await User.findOne({ username });

      // Check if the user exists
      if (!user) {
        throw new Error('Invalid username or password');
      }

      // Check the password
      const isValid = await bcrypt.compare(password, user.password);

      // Check if the password is valid
      if (!isValid) {
        throw new Error('Invalid username or password');
      }

      // Generate a JWT token
      const token = jwt.sign({ userId: user._id, username: user.username }, process.env.SECRET_KEY, {
        expiresIn: '1h'
      });

      // Return the JWT token
      return token;
    } catch (error) {
      // Throw the error
      throw error;
    }
  },

  // Get a user by ID
  getUserById: async (userId) => {
    try {
      // Find the user by ID
      const user = await User.findById(userId);

      // Return the user document
      return user;
    } catch (error) {
      // Throw the error
      throw error;
    }
  },

  // Get a user by username
  getUserByUsername: async (username) => {
    try {
      // Find the user by username
      const user = await User.findOne({ username });

      // Return the user document
      return user;
    } catch (error) {
      // Throw the error
      throw error;
    }
  },

  // Update a user
  updateUser: async (userId, updates) => {
    try {
      // Find the user by ID
      const user = await User.findById(userId);

      // Update the user document
      Object.assign(user, updates);

      // Save the user to the database
      await user.save();

      // Return the updated user document
      return user;
    } catch (error) {
      // Throw the error
      throw error;
    }
  },

  // Delete a user
  deleteUser: async (userId) => {
    try {
      // Find the user by ID
      const user = await User.findById(userId);

      // Delete the user from the database
      await user.remove();

      // Return a success message
      return { message: 'User  deleted successfully' };
    } catch (error) {
      // Throw the error
      throw error;
    }
  }
};

// Export the user service
module.exports = userService;
