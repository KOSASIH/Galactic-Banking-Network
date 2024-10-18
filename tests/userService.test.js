// tests/userService.test.js
const userService = require('../src/user/userService');
const User = require('../src/user/userModel');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

jest.mock('../src/user/userModel');
jest.mock('bcrypt');
jest.mock('jsonwebtoken');

describe('User  Service', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    test('should register a user successfully', async () => {
        const mockUser  = {
            username: 'testuser',
            email: 'test@example.com',
            password: 'password123',
        };

        bcrypt.hash.mockResolvedValue('hashedPassword');
        User.prototype.save.mockResolvedValue(mockUser );

        const user = await userService.registerUser (mockUser .username, mockUser .email, mockUser .password);
        expect(user).toEqual(mockUser );
        expect(bcrypt.hash).toHaveBeenCalledWith(mockUser .password, expect.any(Number));
        expect(User.prototype.save).toHaveBeenCalled();
    });

    test('should throw an error when registering a user fails', async () => {
        User.prototype.save.mockRejectedValue(new Error('Database Error'));

        await expect(userService.registerUser ('testuser', 'test@example.com', 'password123')).rejects.toThrow('Database Error');
    });

    test('should authenticate a user successfully', async () => {
const mockUser  = {
            username: 'testuser',
            email: 'test@example.com',
            password: 'password123',
        };

        bcrypt.compare.mockResolvedValue(true);
        jwt.sign.mockReturnValue('jsonwebtoken');

        const token = await userService.authenticateUser (mockUser .username, mockUser .password);
        expect(token).toBe('jsonwebtoken');
        expect(bcrypt.compare).toHaveBeenCalledWith(mockUser .password, expect.any(String));
        expect(jwt.sign).toHaveBeenCalledWith(expect.any(Object), expect.any(String), expect.any(Object));
    });

    test('should throw an error when authenticating a user fails', async () => {
        bcrypt.compare.mockRejectedValue(new Error('Authentication Failed'));

        await expect(userService.authenticateUser ('testuser', 'password123')).rejects.toThrow('Authentication Failed');
    });
});
