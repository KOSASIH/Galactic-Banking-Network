// Import required modules
const User = require('./dbConfig').User ; // Assuming User model is defined in dbConfig
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// User Controller
class UserController {
  // Method to register a new user
  static async register(req, res) {
    const { username, email, password } = req.body;

    try {
      // Check if the user already exists
      const existingUser  = await User.findOne({ email });
      if (existingUser ) {
        return res.status(400).json({ message: 'User  already exists' });
      }

      // Hash the password
      const hashedPassword = await bcrypt.hash(password, 10);

      // Create a new user
      const newUser  = new User({
        username,
        email,
        password: hashedPassword,
      });

      await newUser .save();
      return res.status(201).json({ message: 'User  registered successfully', user: newUser  });
    } catch (error) {
      console.error('Error registering user:', error);
      return res.status(500).json({ message: 'Failed to register user' });
    }
  }

  // Method to log in a user
  static async login(req, res) {
    const { email, password } = req.body;

    try {
      // Find the user by email
      const user = await User.findOne({ email });
      if (!user) {
        return res.status(404).json({ message: 'User  not found' });
      }

      // Compare the password
      const isMatch = await bcrypt.compare(password, user.password);
      if (!isMatch) {
        return res.status(401).json({ message: 'Invalid credentials' });
      }

      // Generate a JWT token
      const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });

      return res.status(200).json({ message: 'Login successful', token });
    } catch (error) {
      console.error('Error logging in user:', error);
      return res.status(500).json({ message: 'Failed to log in user' });
    }
  }

  // Method to get user profile
  static async getUser Profile(req, res) {
    const userId = req.user.id; // Assuming user ID is stored in req.user after authentication

    try {
      const user = await User.findById(userId).select('-password'); // Exclude password from response
      if (!user) {
        return res.status(404).json({ message: 'User  not found' });
      }
      return res.status(200).json(user);
    } catch (error) {
      console.error('Error retrieving user profile:', error);
      return res.status(500).json({ message: 'Failed to retrieve user profile' });
    }
  }

  // Method to update user profile
  static async updateUser Profile(req, res) {
    const userId = req.user.id; // Assuming user ID is stored in req.user after authentication
    const { username, email } = req.body;

    try {
      const user = await User.findById(userId);
      if (!user) {
        return res.status(404).json({ message: 'User  not found' });
      }

      // Update user details
      user.username = username || user.username;
      user.email = email || user.email;

      await user.save();
      return res.status(200).json({ message: 'User  profile updated successfully', user });
    } catch (error) {
      console.error('Error updating user profile:', error);
      return res.status(500).json({ message: 'Failed to update user profile' });
    }
  }

  // Method to delete a user account
  static async deleteUser Account(req, res) {
    const userId = req.user.id; // Assuming user ID is stored in req.user after authentication

    try {
      const result = await User.deleteOne({ _id: userId });
      if (result.deletedCount === 0) {
        return res.status(404).json({ message: 'User  not found' });
      }
      return res.status(200).json({ message: 'User  account deleted successfully' });
    } catch (error) {
      console.error('Error deleting user account:', error);
      return res.status(500).json({ message: 'Failed to delete user account' });
    }
  }
}

// Export the UserController class
module.exports = UserController;
