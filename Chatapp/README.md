<<<<<<< HEAD
# Advanced Real-Time Chat Application

A comprehensive real-time chat application built with Node.js, Socket.IO, and MongoDB featuring advanced messaging capabilities, voice/video calls, and modern UI.

## ✨ Features

### 🔐 Authentication & User Management
- **User Registration & Login**: Secure authentication with JWT tokens
- **Auto-login**: Persistent sessions with JWT token storage
- **User Profiles**: Username, email, and avatar support
- **Online Status**: Real-time online/offline status tracking
- **Last Seen**: Track when users were last active
- **Logout**: Secure session termination

### 💬 Messaging Features
- **Real-time Messaging**: Instant message delivery using Socket.IO
- **Room-based Chat**: Join and chat in different rooms
- **Private Messaging**: One-on-one private conversations
- **Message Persistence**: All messages stored in MongoDB database
- **Message Types**: Support for text, images, files, and voice messages
- **Message Editing**: Edit your own messages with edit history
- **Message Deletion**: Soft delete messages (mark as deleted)
- **Message Reactions**: React to messages with emojis (👍, ❤️, 😊, 😂, 😮, 😢, 😡)
- **Message Replies**: Reply to specific messages
- **Message Search**: Search through message history
- **Message Timestamps**: See when messages were sent/edited
- **Pagination**: Load messages in pages for better performance

### 📁 File Sharing
- **File Upload**: Share various file types (images, documents, videos, audio)
- **Supported Formats**: 
  - Images: JPEG, JPG, PNG, GIF
  - Documents: PDF, DOC, DOCX, TXT
  - Audio: MP3, WAV, MPEG
  - Video: MP4, AVI, MOV, WMV, FLV, WEBM, MKV, 3GP
- **File Size Limit**: Up to 500MB for video files
- **File Preview**: View file names, sizes, and types
- **Secure Storage**: Files stored in uploads directory with unique names

### 📞 Voice & Video Calls
- **WebRTC Integration**: Peer-to-peer voice and video calls
- **Room-based Calls**: Start calls in chat rooms
- **Private Calls**: Direct voice/video calls between users
- **Call Signaling**: Complete WebRTC signaling implementation
- **Call Controls**: Accept, reject, and end calls
- **ICE Candidate Handling**: NAT traversal support

### 🎨 User Interface
- **Modern Design**: Beautiful, responsive UI with gradient backgrounds
- **Real-time Updates**: Live typing indicators and online status
- **Online Users List**: See who's currently online
- **Typing Indicators**: Know when someone is typing
- **Message Status**: Visual indicators for message states
- **Responsive Layout**: Works on desktop and mobile devices
- **Smooth Animations**: CSS animations and transitions
- **Dark/Light Theme**: Modern color scheme

### 🔍 Advanced Features
- **Message Search**: Search through message content
- **User Search**: Find users for private messaging
- **Real-time Notifications**: Instant updates for new messages
- **Connection Management**: Automatic reconnection handling
- **Error Handling**: Comprehensive error handling and user feedback
- **Performance Optimization**: Pagination and efficient data loading

## 🛠️ Technologies Used

### Backend
- **Node.js**: JavaScript runtime environment
- **Express.js**: Web application framework
- **Socket.IO**: Real-time bidirectional communication
- **MongoDB**: NoSQL database for data persistence
- **Mongoose**: MongoDB object modeling for Node.js
- **JWT**: JSON Web Tokens for authentication
- **bcryptjs**: Password hashing and verification
- **multer**: File upload handling
- **cors**: Cross-origin resource sharing
- **dotenv**: Environment variable management

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with gradients and animations
- **JavaScript (ES6+)**: Vanilla JavaScript with modern features
- **Font Awesome**: Icon library
- **WebRTC API**: For voice and video calls

### Database
- **MongoDB Atlas**: Cloud-hosted MongoDB database
- **Mongoose Schemas**: Structured data models
- **Indexing**: Optimized database queries
- **Data Relationships**: User and message associations

### Real-time Communication
- **Socket.IO**: WebSocket implementation
- **WebRTC**: Peer-to-peer communication
- **ICE Candidates**: Network traversal
- **Signaling Server**: Call coordination

## 📁 Project Structure

```
websocket/
├── index.js              # Main server file with Socket.IO setup
├── index.html            # Frontend application (3200+ lines)
├── package.json          # Dependencies and scripts
├── config.env            # Environment variables
├── models/
│   ├── User.js          # User model with online status
│   └── Message.js       # Message model with reactions & replies
├── routes/
│   ├── auth.js          # Authentication routes
│   └── messages.js      # Message CRUD operations
├── middleware/
│   ├── auth.js          # JWT authentication middleware
│   └── upload.js        # File upload middleware
├── uploads/             # File storage directory
└── README.md            # This file
```

## 🚀 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/logout` - Logout user

### Messages
- `GET /api/messages/room/:room` - Get room messages with pagination
- `POST /api/messages/send` - Send text message
- `PUT /api/messages/:messageId` - Edit message
- `DELETE /api/messages/:messageId` - Delete message
- `POST /api/messages/:messageId/reactions` - Add reaction to message
- `GET /api/messages/search/:query` - Search messages
- `POST /api/messages/upload` - Upload file
- `GET /api/messages/private/:otherUserId` - Get private messages

### Socket.IO Events
- `join-room` - Join a chat room
- `message` - Send a message
- `private-message` - Send private message
- `join-private-chat` - Join private chat
- `typing` - Typing indicator
- `new message` - Receive new message
- `user-joined` - User joined room
- `online-users` - List of online users
- `user-typing` - User typing indicator
- `message-updated` - Message updated (edit/reaction)
- `call-request` - Request voice/video call
- `private-call-request` - Request private call
- `call-offer` - WebRTC offer
- `call-answer` - WebRTC answer
- `ice-candidate` - ICE candidate exchange
- `call-ended` - Call ended
- `call-rejected` - Call rejected

## ⚙️ Installation & Setup

### Prerequisites
- Node.js (v14 or higher)
- MongoDB Atlas account (or local MongoDB)
- npm or yarn

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd websocket
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `config.env` file in the root directory:
   ```env
   JWT_SECRET=your-secret-key-here
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/chatapp
   PORT=3000
   ```

4. **Start the server**
   ```bash
   node index.js
   ```

5. **Access the application**
   - Open `http://localhost:3000` in your browser
   - Register a new account or login with existing credentials

## 🎯 Usage Guide

### Getting Started
1. **Register/Login**: Create an account or login with existing credentials
2. **Join Room**: Enter a room name and click "Join"
3. **Start Chatting**: Type messages and press Enter or click Send
4. **File Sharing**: Click the attachment icon to upload files
5. **Private Messages**: Click on online users to start private conversations

### Advanced Features
- **Voice/Video Calls**: Use the call buttons for room or private calls
- **Message Reactions**: Click on messages to add emoji reactions
- **Message Replies**: Click reply to respond to specific messages
- **Message Search**: Use the search function to find specific messages
- **Message Editing**: Click the edit button on your messages

## 🔧 Configuration

### Environment Variables
- `JWT_SECRET`: Secret key for JWT token signing
- `MONGODB_URI`: MongoDB connection string
- `PORT`: Server port (default: 3000)

### File Upload Settings
- Maximum file size: 500MB
- Supported file types: Images, documents, audio, video
- Storage location: `uploads/` directory

### Database Configuration
- MongoDB Atlas with connection retry logic
- Automatic reconnection on connection loss
- Optimized indexes for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the ISC License.

## 🆘 Support

If you encounter any issues or have questions:
- Check the console for error messages
- Verify your MongoDB connection
- Ensure all environment variables are set correctly
- Open an issue on GitHub with detailed information

## 🔮 Future Enhancements

- Push notifications
- Message encryption
- Group chat management
- User blocking/muting
- Message forwarding
- Advanced file preview
- Screen sharing
- Chat backup/export
- Multi-language support
- Mobile app development 
=======
# Chatapp
>>>>>>> 4d804c9acc75c188ac4c87a07d33ca81368e3480
