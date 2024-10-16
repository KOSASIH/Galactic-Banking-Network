// Import required modules
const nodemailer = require('nodemailer');
const twilio = require('twilio');
const pusher = require('pusher');
const { Notification } = require('./notificationModel');

// Define the NotificationService class
class NotificationService {
  // Method to send an email notification
  async sendEmailNotification(to, subject, body) {
    try {
      const transporter = nodemailer.createTransport({
        host: 'smtp.example.com',
        port: 587,
        secure: false, // or 'STARTTLS'
        auth: {
          user: 'username',
          pass: 'password',
        },
      });

      const mailOptions = {
        from: 'from@example.com',
        to,
        subject,
        text: body,
      };

      await transporter.sendMail(mailOptions);
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  // Method to send an SMS notification using Twilio
  async sendSMSNotification(to, body) {
    try {
      const accountSid = 'your_account_sid';
      const authToken = 'your_auth_token';
      const client = new twilio(accountSid, authToken);

      const message = await client.messages
        .create({
          body,
          from: 'your_twilio_number',
          to,
        })
        .done();

      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  // Method to send a push notification using Pusher
  async sendPushNotification(userId, title, message) {
    try {
      const pusherInstance = new pusher({
        appId: 'your_app_id',
        key: 'your_app_key',
        secret: 'your_app_secret',
        cluster: 'your_cluster',
      });

      await pusherInstance.trigger('notifications', 'new-notification', {
        userId,
        title,
        message,
      });

      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  // Method to create a new notification in the database
  async createNotification(userId, title, message) {
    try {
      const notification = new Notification({
        userId,
        title,
        message,
        createdAt: Date.now(),
      });

      await notification.save();
      return true;
    } catch (error) {
      console.error(error);
      return false;
    }
  }

  // Method to get all notifications for a user
  async getNotificationsForUser (userId) {
    try {
      const notifications = await Notification.find({ userId });
      return notifications;
    } catch (error) {
      console.error(error);
      return [];
    }
  }
}

// Export the NotificationService class
module.exports = NotificationService;
