// Import required modules
const mongoose = require('mongoose');
const Schema = mongoose.Schema;
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

// Define the User schema
const userSchema = new Schema({
  _id: { type: String, required: true, unique: true },
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: { type: String, required: true, enum: ['admin', 'user'] },
  profilePicture: { type: String },
  bio: { type: String },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now },
});

// Define the User model
const User = mongoose.model('User ', userSchema);

// Export the User model
module.exports = User;

// Define the User methods
User .methods = {
  // Method to generate a JWT token
  generateToken: function() {
    return jwt.sign({ id: this._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
  },

  // Method to compare passwords
  comparePassword: function(password) {
    return bcrypt.compare(password, this.password);
  },
};

// Define the User static methods
User .statics = {
  // Method to find a user by ID
  findById: function(id) {
    return this.findOne({ _id: id });
  },

  // Method to find a user by username
  findByUsername: function(username) {
    return this.findOne({ username });
  },

  // Method to find a user by email
  findByEmail: function(email) {
    return this.findOne({ email });
  },
};

// Define the User hooks
User .pre('save', function(next) {
  // Hash the password before saving
  bcrypt.hash(this.password, 10, (err, hash) => {
    if (err) {
      return next(err);
    }
    this.password = hash;
    next();
  });
});

User .post('save', function(doc) {
  // Send a welcome email to the user after saving
  console.log(`User  saved: ${doc._id}`);
});

// Export the User model with methods and static methods
module.exports = User;
