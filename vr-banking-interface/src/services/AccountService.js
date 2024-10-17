// src/services/AccountService.js

class AccountService {
    static async getAccountDetails(userId) {
        // Simulate an API call to fetch account details
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    balance: 1000,
                    holderName: 'John Doe',
                    accountId: userId,
                });
            }, 1000); // Simulate network delay
        });
    }
}

export default AccountService;
