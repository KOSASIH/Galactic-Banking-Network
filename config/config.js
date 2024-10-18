// config/config.js
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

const config = {
    app: {
        port: process.env.PORT || 3000,
        env: process.env.NODE_ENV || 'development',
    },
    db: {
        uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/your_database_name',
        options: {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        },
    },
    jwt: {
        secret: process.env.JWT_SECRET || 'your_jwt_secret',
        expiresIn: process.env.JWT_EXPIRES_IN || '1h',
    },
    logging: {
        level: process.env.LOGGING_LEVEL || 'info',
    },
};

module.exports = config;
