// Import required modules
const mongoose = require('mongoose');
const { User } = require('./userModel');
const { UserService } = require('./userService');
const bcrypt = require('bcrypt');

// Define the test suite for UserService
describe('UserService', () => {
  let userService;

  // Before each test, connect to the database and create a new user service
  beforeEach(async () => {
    await mongoose.connect('mongodb://localhost/test', { useNewUrlParser: true, useUnifiedTopology: true });
    userService = new UserService();

    // Create test user
    await User.create({
      _id: 'user_id_1',
      email: 'user1@example.com',
      password: await bcrypt.hash('password123', 10),
    });
  });

  // After each test, disconnect from the database and clean up
  afterEach(async () => {
    await User.deleteMany({});
    await mongoose.disconnect();
  });

  // Test 1: Create a new user
  it('should create a new user', async () => {
    const user = await userService.createUser('user2@example.com', 'password123');

    expect(user).toBeDefined();
    expect(user.email).toBe('user2@example.com');
  });

  // Test 2: Get a user by ID
  it('should get a user by ID', async () => {
    const user = await userService.getUserById('user_id_1');

    expect(user).toBeDefined();
    expect(user.id).toBe('user_id_1');
  });

  // Test 3: Get a user by email
  it('should get a user by email', async () => {
    const user = await userService.getUserByEmail('user1@example.com');

    expect(user).toBeDefined();
    expect(user.email).toBe('user1@example.com');
  });

  // Test 4: Update a user's email
  it('should update a user\'s email', async () => {
    const user = await userService.updateUser Email('user_id_1', 'newemail@example.com');

    expect(user).toBeDefined();
    expect(user.email).toBe('newemail@example.com');
  });

  // Test 5: Update a user's password
  it('should update a user\'s password', async () => {
    const user = await userService.updateUser Password('user_id_1', 'newpassword123');

    expect(user).toBeDefined();
    const isValidPassword = await bcrypt.compare('newpassword123', user.password);
    expect(isValidPassword).toBe(true);
  });

  // Test 6: Handle invalid user ID during user retrieval
  it('should throw an error for invalid user ID', async () => {
    await expect(userService.getUserById('invalid_user_id')).rejects.toThrow('User  not found');
  });

  // Test 7: Handle duplicate email during user creation
  it('should throw an error for duplicate email', async () => {
    await expect(userService.createUser('user1@example.com', 'password123')).rejects.toThrow('Email already exists');
  });
});
