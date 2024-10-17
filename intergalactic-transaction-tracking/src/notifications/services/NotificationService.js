// notifications/services/NotificationService.js

const nodemailer = require('nodemailer');
const twilio = require('twilio');

class NotificationService {
    constructor() {
        // Configure email transport
        this.emailTransporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.EMAIL_USER, // Your email
                pass: process.env.EMAIL_PASS  // Your email password
            }
        });

        // Configure Twilio for SMS
        this.twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH_TOKEN);
    }

    // Send email notification
    async sendEmail(to, subject, text) {
        const mailOptions = {
            from: process.env.EMAIL_USER,
            to,
            subject,
            text
        };

        try {
            const info = await this.emailTransporter.sendMail(mailOptions);
            console.log('Email sent:', info.response);
        } catch (error) {
            throw new Error(`Error sending email: ${error.message}`);
        }
    }

    // Send SMS notification
    async sendSMS(to, message) {
        try {
            const messageResponse = await this.twilioClient.messages.create({
                body: message,
                from: process.env.TWILIO_PHONE_NUMBER, // Your Twilio phone number
                to
            });
            console.log('SMS sent:', messageResponse.sid);
        } catch (error) {
            throw new Error(`Error sending SMS: ${error.message}`);
        }
    }

    // Send notification based on type
    async sendNotification(type, to, subjectOrMessage, text) {
        try {
            if (type === 'email') {
                await this.sendEmail(to, subjectOrMessage, text);
            } else if (type === 'sms') {
                await this.sendSMS(to, subjectOrMessage);
            } else {
                throw new Error('Unsupported notification type');
            }
        } catch (error) {
            console.error(`Error sending notification: ${error.message}`);
        }
    }
}

module.exports = new NotificationService();
