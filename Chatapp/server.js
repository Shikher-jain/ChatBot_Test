// Load environment variables - works for both local and Vercel
if (process.env.NODE_ENV !== 'production') {
    require('dotenv').config({ path: './config.env' });
} else {
    // For production (Vercel), environment variables are set directly
    console.log('🌐 Running in production mode - using Vercel environment variables');
}

const express = require('express');
const path = require('path');
const mongoose = require('mongoose');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const User = require('./models/User');

const app = express();
const { Server } = require('socket.io');
const http = require('http');
const server = http.createServer(app);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// MongoDB Connection
const MONGODB_URI = process.env.MONGODB_URI;

if (!MONGODB_URI) {
    console.error('MONGODB_URI environment variable is not set');
    process.exit(1);
}

// Connection retry logic
let retryCount = 0;
const maxRetries = 3;

function connectToMongoDB() {
    console.log(`🔄 Attempting to connect to MongoDB Atlas... (Attempt ${retryCount + 1}/${maxRetries})`);
    
    mongoose.connect(MONGODB_URI, {
        serverSelectionTimeoutMS: 5000,
        socketTimeoutMS: 45000,
        connectTimeoutMS: 10000,
        maxPoolSize: 10,
        retryWrites: true,
        w: 'majority'
    })
    .then(() => {
        console.log('✅ Connected to MongoDB Atlas successfully!');
        console.log('📊 Database: chatapp');
        retryCount = 0; // Reset retry count on success
    })
    .catch(err => {
        retryCount++;
        console.error(`❌ MongoDB connection error (Attempt ${retryCount}/${maxRetries}):`, err.message);
        
        if (retryCount < maxRetries) {
            console.log(`🔄 Retrying in 3 seconds...`);
            setTimeout(connectToMongoDB, 3000);
        } else {
            console.log('🔧 Please check:');
            console.log('   1. Your internet connection');
            console.log('   2. MongoDB Atlas IP whitelist (add your IP)');
            console.log('   3. Connection string in config.env');
            console.log('   4. Atlas cluster is running');
            console.log('   5. Try restarting the server');
            process.exit(1);
        }
    });
}

// Start connection
connectToMongoDB();

// Routes
const authRoutes = require('./routes/auth');
const messagesRoutes = require('./routes/messages');
app.use('/api/auth', authRoutes);
app.use('/api/messages', messagesRoutes);

// Serve HTML
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Socket.IO with authentication
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Store online users
const onlineUsers = new Map();

app.set('io', io); // Make io accessible in routes
app.set('onlineUsers', onlineUsers); // Make onlineUsers accessible in routes

// Socket authentication middleware
io.use(async (socket, next) => {
    try {
        const token = socket.handshake.auth.token;
        if (!token) {
            return next(new Error('Authentication error'));
        }

        const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
        const user = await User.findById(decoded.userId);
        
        if (!user) {
            return next(new Error('User not found'));
        }

        socket.userId = user._id;
        socket.username = user.username;
        next();
    } catch (error) {
        next(new Error('Authentication error'));
    }
});

io.on("connection", async (socket) => {
    console.log(`👤 User ${socket.username} connected with socket ID: ${socket.id}`);

    // Update user online status and socketId
    await User.findByIdAndUpdate(socket.userId, {
        isOnline: true,
        lastSeen: new Date(),
        socketId: socket.id // Save the socket ID
    });

    // Add to online users map
    onlineUsers.set(socket.userId.toString(), {
        id: socket.userId,
        username: socket.username,
        socketId: socket.id
    });

    // Emit online users list
    io.emit('online-users', Array.from(onlineUsers.values()));

    // Join room
    socket.on('join-room', (room) => {
        socket.join(room);
        socket.to(room).emit('user-joined', {
            username: socket.username,
            message: `${socket.username} joined the room`
        });
        console.log(`🏠 ${socket.username} joined room: ${room}`);
    });

    // Send message
    socket.on('message', ({ room, msg }) => {
        const messageData = {
            message: msg,
            username: socket.username,
            userId: socket.userId,
            timestamp: new Date(),
            room: room
        };

        // Emit to all in room (including sender)
        io.to(room).emit('new message', messageData);
        io.to(socket.id).emit('new message', messageData);
        console.log(`💬 ${socket.username} sent message in room ${room}: ${msg}`);
    });

    // Send private message
    socket.on('private-message', ({ receiverId, msg }) => {
        console.log(`🔍 Private message debug - Receiver ID: ${receiverId}, Type: ${typeof receiverId}`);
        
        // Ensure receiverId is a string for Map lookup
        const receiverIdStr = receiverId.toString();
        
        const messageData = {
            message: msg,
            username: socket.username,
            userId: socket.userId,
            timestamp: new Date(),
            room: `private_${socket.userId}_${receiverIdStr}`,
            isPrivate: true,
            receiverId: receiverIdStr
        };

        // Send to receiver if online
        const receiverSocket = onlineUsers.get(receiverIdStr);
        if (receiverSocket) {
            console.log(`📤 Sending to receiver: ${receiverSocket.username} (${receiverSocket.socketId})`);
            io.to(receiverSocket.socketId).emit('private-message', messageData);
        } else {
            console.log(`⚠️ Receiver ${receiverIdStr} not found in online users`);
        }

        // Always send to sender
        console.log(`📤 Sending to sender: ${socket.username} (${socket.id})`);
        io.to(socket.id).emit('private-message', messageData);
        
        console.log(`💬 ${socket.username} sent private message to ${receiverIdStr}: ${msg}`);
    });

    // Join private chat
    socket.on('join-private-chat', ({ otherUserId }) => {
        const otherUserIdStr = otherUserId.toString();
        const privateRoom = `private_${socket.userId}_${otherUserIdStr}`;
        socket.join(privateRoom);
        console.log(`🔒 ${socket.username} joined private chat with ${otherUserIdStr} (Room: ${privateRoom})`);
    });

    // Typing indicator
    socket.on('typing', ({ room, isTyping }) => {
        socket.to(room).emit('user-typing', {
            username: socket.username,
            isTyping: isTyping
        });
    });

    // WebRTC Signaling Events
    // Call request (room-based)
    socket.on('call-request', (data) => {
        console.log(`📞 ${socket.username} requesting ${data.type} call in room ${data.room}`);
        socket.to(data.room).emit('call-request', {
            type: data.type,
            room: data.room,
            caller: socket.username
        });
    });

    // Private call request
    socket.on('private-call-request', (data) => {
        console.log(`📞 ${socket.username} requesting private ${data.type} call to user ${data.targetUserId}`);
        const targetUser = onlineUsers.get(data.targetUserId.toString());
        if (targetUser) {
            io.to(targetUser.socketId).emit('private-call-request', {
                type: data.type,
                targetUserId: data.targetUserId,
                caller: socket.username
            });
        }
    });

    // Call offer
    socket.on('call-offer', (data) => {
        console.log(`📞 ${socket.username} sending call offer`);
        if (data.room) {
            socket.to(data.room).emit('call-offer', {
                offer: data.offer,
                type: data.type,
                room: data.room,
                caller: socket.username
            });
        } else if (data.targetUserId) {
            const targetUser = onlineUsers.get(data.targetUserId.toString());
            if (targetUser) {
                io.to(targetUser.socketId).emit('call-offer', {
                    offer: data.offer,
                    type: data.type,
                    targetUserId: data.targetUserId,
                    caller: socket.username
                });
            }
        }
    });

    // Call answer
    socket.on('call-answer', (data) => {
        console.log(`📞 ${socket.username} sending call answer`);
        if (data.room) {
            socket.to(data.room).emit('call-answer', {
                answer: data.answer,
                room: data.room,
                caller: socket.username
            });
        } else if (data.targetUserId) {
            const targetUser = onlineUsers.get(data.targetUserId.toString());
            if (targetUser) {
                io.to(targetUser.socketId).emit('call-answer', {
                    answer: data.answer,
                    targetUserId: data.targetUserId,
                    caller: socket.username
                });
            }
        }
    });

    // ICE candidate
    socket.on('ice-candidate', (data) => {
        console.log(`📞 ${socket.username} sending ICE candidate`);
        if (data.room) {
            socket.to(data.room).emit('ice-candidate', {
                candidate: data.candidate,
                room: data.room,
                caller: socket.username
            });
        } else if (data.targetUserId) {
            const targetUser = onlineUsers.get(data.targetUserId.toString());
            if (targetUser) {
                io.to(targetUser.socketId).emit('ice-candidate', {
                    candidate: data.candidate,
                    targetUserId: data.targetUserId,
                    caller: socket.username
                });
            }
        }
    });

    // Call ended
    socket.on('call-ended', (data) => {
        console.log(`📞 ${socket.username} ended call`);
        if (data.room) {
            socket.to(data.room).emit('call-ended', {
                room: data.room
            });
        } else if (data.targetUserId) {
            const targetUser = onlineUsers.get(data.targetUserId.toString());
            if (targetUser) {
                io.to(targetUser.socketId).emit('call-ended', {
                    targetUserId: data.targetUserId
                });
            }
        }
    });

    // Call rejected
    socket.on('call-rejected', (data) => {
        console.log(`📞 ${socket.username} rejected call in room ${data.room}`);
        socket.to(data.room).emit('call-rejected', {
            room: data.room,
            caller: data.caller
        });
    });

    // Private call rejected
    socket.on('private-call-rejected', (data) => {
        console.log(`📞 ${socket.username} rejected private call from ${data.caller}`);
        const targetUser = onlineUsers.get(data.targetUserId.toString());
        if (targetUser) {
            io.to(targetUser.socketId).emit('private-call-rejected', {
                targetUserId: data.targetUserId,
                caller: data.caller
            });
        }
    });

    // Private call accepted
    socket.on('private-call-accepted', (data) => {
        console.log(`📞 ${socket.username} accepted private call from ${data.caller}`);
        const targetUser = onlineUsers.get(data.targetUserId.toString());
        if (targetUser) {
            io.to(targetUser.socketId).emit('private-call-accepted', {
                targetUserId: data.targetUserId,
                caller: data.caller,
                type: data.type
            });
        }
    });

    // Disconnect
    socket.on("disconnect", async () => {
        console.log(`👋 User ${socket.username} disconnected`);
        
        // Update user offline status
        await User.findByIdAndUpdate(socket.userId, {
            isOnline: false,
            lastSeen: new Date(),
            socketId: null // Clear socket ID on disconnect
        });

        // Remove from online users
        onlineUsers.delete(socket.userId.toString());
        
        // Emit updated online users list
        io.emit('online-users', Array.from(onlineUsers.values()));
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`🚀 Server listening at port ${PORT}`);
    console.log(`🌐 Open http://localhost:${PORT} in your browser`);
    console.log(`📱 Environment: ${process.env.NODE_ENV || 'development'}`);
});