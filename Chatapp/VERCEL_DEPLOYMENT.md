# Vercel Deployment Guide for Chat App

## 🚨 Important: Fix MongoDB Connection Issue

Your app is failing on Vercel because it's trying to connect to localhost MongoDB. Here's how to fix it:

## 🔧 Step 1: Set Up MongoDB Atlas (Cloud Database)

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free account or sign in
3. Create a new cluster (free tier works fine)
4. Get your connection string

## 🔑 Step 2: Set Environment Variables in Vercel

1. Go to your Vercel dashboard
2. Select your project
3. Go to **Settings** → **Environment Variables**
4. Add these variables:

```
MONGODB_URI=mongodb+srv://yourusername:yourpassword@yourcluster.mongodb.net/chatapp?retryWrites=true&w=majority
JWT_SECRET=your-super-secret-jwt-key-here
NODE_ENV=production
PORT=3000
CORS_ORIGIN=*
```

## 📝 Step 3: Update MongoDB Atlas Settings

1. In MongoDB Atlas, go to **Network Access**
2. Click **Add IP Address**
3. Click **Allow Access from Anywhere** (for development)
4. Or add Vercel's IP ranges

## 🚀 Step 4: Redeploy

1. Push your changes to GitHub
2. Vercel will automatically redeploy
3. Check the logs for successful MongoDB connection

## ✅ What This Fixes

- ❌ **Before**: `mongodb://localhost:27017/chatapp` (only works locally)
- ✅ **After**: `mongodb+srv://...` (works on Vercel)

## 🔍 Troubleshooting

If you still have issues:

1. Check Vercel logs for MongoDB connection errors
2. Verify your MongoDB Atlas connection string
3. Make sure your MongoDB Atlas cluster is running
4. Check if your IP is whitelisted in MongoDB Atlas

## 📱 Test Your App

After deployment, try:
1. Opening your Vercel URL
2. Registering a new user
3. Sending messages
4. Making voice/video calls

## 💡 Pro Tips

- Use MongoDB Atlas free tier (512MB) for development
- Set up proper IP whitelisting for production
- Use strong JWT secrets
- Monitor your MongoDB Atlas usage

## 🆘 Need Help?

- Check Vercel deployment logs
- Verify MongoDB Atlas connection
- Ensure all environment variables are set correctly
