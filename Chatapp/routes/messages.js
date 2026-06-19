const express = require('express');
const Message = require('../models/Message');
const User = require('../models/User');
const auth = require('../middleware/auth');
const upload = require('../middleware/upload');

const router = express.Router();

// Get messages for a room
router.get('/room/:room', auth, async (req, res) => {
    try {
        const { room } = req.params;
        const { page = 1, limit = 50 } = req.query;
        
        const messages = await Message.find({
            room: room,
            isDeleted: false
        })
        .populate('sender', 'username avatar')
        .populate('reactions.user', 'username')
        .populate('replyTo')
        .sort({ createdAt: -1 })
        .limit(limit * 1)
        .skip((page - 1) * limit)
        .exec();

        const count = await Message.countDocuments({
            room: room,
            isDeleted: false
        });

        res.json({
            messages: messages.reverse(),
            totalPages: Math.ceil(count / limit),
            currentPage: page,
            totalMessages: count
        });
    } catch (error) {
        console.error('Error fetching messages:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Send text message
router.post('/send', auth, async (req, res) => {
    try {
        const { content, room, receiverId, replyToId, isPrivate = false } = req.body;

        if (!content || (!room && !receiverId)) {
            return res.status(400).json({ error: 'Content and room/receiver are required' });
        }

        const messageData = {
            sender: req.user._id,
            content: content,
            messageType: 'text',
            isPrivate: isPrivate
        };

        if (isPrivate && receiverId) {
            messageData.receiver = receiverId;
            messageData.room = `private_${req.user._id}_${receiverId}`;
        } else {
            messageData.room = room;
        }

        if (replyToId) {
            messageData.replyTo = replyToId;
        }

        const message = new Message(messageData);
        await message.save();

        const populatedMessage = await Message.findById(message._id)
            .populate('sender', 'username avatar')
            .populate('receiver', 'username avatar')
            .populate('replyTo')
            .populate('reactions.user', 'username');

        res.status(201).json(populatedMessage);
    } catch (error) {
        console.error('Error sending message:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Edit message
router.put('/:messageId', auth, async (req, res) => {
    try {
        const { messageId } = req.params;
        const { content } = req.body;

        if (!content) {
            return res.status(400).json({ error: 'Content is required' });
        }

        const message = await Message.findById(messageId);
        
        if (!message) {
            return res.status(404).json({ error: 'Message not found' });
        }

        if (message.sender.toString() !== req.user._id.toString()) {
            return res.status(403).json({ error: 'You can only edit your own messages' });
        }

        message.content = content;
        message.isEdited = true;
        message.editedAt = new Date();
        await message.save();

        const updatedMessage = await Message.findById(messageId)
            .populate('sender', 'username avatar')
            .populate('receiver', 'username avatar')
            .populate('reactions.user', 'username')
            .populate('replyTo');

        res.json(updatedMessage);
    } catch (error) {
        console.error('Error editing message:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Delete message
router.delete('/:messageId', auth, async (req, res) => {
    try {
        const { messageId } = req.params;

        const message = await Message.findById(messageId);
        
        if (!message) {
            return res.status(404).json({ error: 'Message not found' });
        }

        if (message.sender.toString() !== req.user._id.toString()) {
            return res.status(403).json({ error: 'You can only delete your own messages' });
        }

        message.isDeleted = true;
        message.deletedAt = new Date();
        await message.save();

        res.json({ message: 'Message deleted successfully' });
    } catch (error) {
        console.error('Error deleting message:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Add reaction to message
router.post('/:messageId/reactions', auth, async (req, res) => {
    try {
        const { messageId } = req.params;
        const { emoji } = req.body;

        if (!emoji) {
            return res.status(400).json({ error: 'Emoji is required' });
        }

        const message = await Message.findById(messageId);
        
        if (!message) {
            return res.status(404).json({ error: 'Message not found' });
        }

        // Remove existing reaction from this user
        message.reactions = message.reactions.filter(
            reaction => reaction.user.toString() !== req.user._id.toString()
        );

        // Add new reaction
        message.reactions.push({
            user: req.user._id,
            emoji: emoji
        });

        await message.save();

        const updatedMessage = await Message.findById(messageId)
            .populate('sender', 'username avatar')
            .populate('receiver', 'username avatar')
            .populate('reactions.user', 'username')
            .populate('replyTo');

        // Emit socket event for real-time reaction update
        const io = req.app.get('io');
        if (message.isPrivate) {
            // For private messages, emit to both sender and receiver
            const sender = req.app.get('onlineUsers').get(message.sender.toString());
            const receiver = req.app.get('onlineUsers').get(message.receiver.toString());
            
            if (sender && sender.socketId) {
                io.to(sender.socketId).emit('message-updated', updatedMessage);
            }
            if (receiver && receiver.socketId) {
                io.to(receiver.socketId).emit('message-updated', updatedMessage);
            }
        } else {
            // For room messages, emit to the room
            io.to(message.room).emit('message-updated', updatedMessage);
        }

        res.json(updatedMessage);
    } catch (error) {
        console.error('Error adding reaction:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Search messages
router.get('/search/:query', auth, async (req, res) => {
    try {
        const { query } = req.params;
        const { room, limit = 20 } = req.query;

        let searchQuery = {
            content: { $regex: query, $options: 'i' },
            isDeleted: false
        };

        if (room) {
            searchQuery.room = room;
        }

        const messages = await Message.find(searchQuery)
            .populate('sender', 'username avatar')
            .populate('receiver', 'username avatar')
            .populate('reactions.user', 'username')
            .populate('replyTo')
            .sort({ createdAt: -1 })
            .limit(parseInt(limit))
            .exec();

        res.json(messages);
    } catch (error) {
        console.error('Error searching messages:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

// Upload File
router.post('/upload', auth, upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded.' });
        }

        const { room, receiverId, isPrivate = false } = req.body;
        
        let messageType = 'file';
        if (req.file.mimetype.startsWith('image/')) {
            messageType = 'image';
        } else if (req.file.mimetype.startsWith('audio/')) {
            messageType = 'voice';
        }

        const messageData = {
            sender: req.user._id,
            content: req.file.filename,
            messageType: messageType,
            fileUrl: `/uploads/${req.file.filename}`,
            fileName: req.file.originalname,
            fileSize: req.file.size,
            isPrivate: isPrivate
        };
        
        if (isPrivate && receiverId) {
            messageData.receiver = receiverId;
            messageData.room = `private_${req.user._id}_${receiverId}`;
        } else {
            messageData.room = room;
        }

        const message = new Message(messageData);
        await message.save();

        const populatedMessage = await Message.findById(message._id)
            .populate('sender', 'username avatar');
            
        // Emit socket event to the room or private chat
        const io = req.app.get('io');
        const onlineUsers = req.app.get('onlineUsers');

        if (message.isPrivate) {
            // Get receiver's socketId from the in-memory map
            const receiver = onlineUsers.get(message.receiver.toString());
            if (receiver && receiver.socketId) {
                io.to(receiver.socketId).emit('new message', populatedMessage);
            }
            // Also send to sender
            const sender = onlineUsers.get(req.user._id.toString());
            if (sender && sender.socketId) {
                io.to(sender.socketId).emit('new message', populatedMessage);
            }
        } else {
            io.to(message.room).emit('new message', populatedMessage);
        }

        res.status(201).json(populatedMessage);

    } catch (error) {
        console.error('File upload error:', error);
        res.status(500).json({ error: 'Server error during file upload.' });
    }
});

// Get private messages between two users
router.get('/private/:otherUserId', auth, async (req, res) => {
    try {
        const { otherUserId } = req.params;
        const { page = 1, limit = 50 } = req.query;
        
        // Create both possible room names (user1_user2 and user2_user1)
        const room1 = `private_${req.user._id}_${otherUserId}`;
        const room2 = `private_${otherUserId}_${req.user._id}`;
        
        const messages = await Message.find({
            room: { $in: [room1, room2] },
            isDeleted: false,
            isPrivate: true
        })
        .populate('sender', 'username avatar')
        .populate('receiver', 'username avatar')
        .populate('reactions.user', 'username')
        .populate('replyTo')
        .sort({ createdAt: -1 })
        .limit(limit * 1)
        .skip((page - 1) * limit)
        .exec();

        const count = await Message.countDocuments({
            room: { $in: [room1, room2] },
            isDeleted: false,
            isPrivate: true
        });

        res.json({
            messages: messages.reverse(),
            totalPages: Math.ceil(count / limit),
            currentPage: page,
            totalMessages: count
        });
    } catch (error) {
        console.error('Error fetching private messages:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

module.exports = router; 