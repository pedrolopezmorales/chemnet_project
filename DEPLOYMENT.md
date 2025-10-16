# ChemNet Deployment Guide

## Prerequisites
1. GitHub account
2. Vercel account (for Next.js)
3. Railway account (for Django API)

## Step 1: Push to GitHub

### For Django Backend:
```bash
cd C:\Users\pelom\chemnet_project
git init
git add .
git commit -m "Initial Django backend commit"
git branch -M main
git remote add origin https://github.com/yourusername/chemnet-backend.git
git push -u origin main
```

### For Next.js Frontend:
```bash
cd C:\Users\pelom\chemnet-frontend
git init
git add .
git commit -m "Initial Next.js frontend commit"  
git branch -M main
git remote add origin https://github.com/yourusername/chemnet-frontend.git
git push -u origin main
```

## Step 2: Deploy Django on Railway

1. Go to https://railway.app
2. Sign up/login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your chemnet-backend repository
5. Railway will auto-detect Django and deploy
6. Set environment variables:
   - `DEBUG=False`
   - `SECRET_KEY=your-secret-key-here`
7. Note your Railway app URL (e.g., https://chemnet-backend-production.railway.app)

## Step 3: Deploy Next.js on Vercel

1. Go to https://vercel.com
2. Sign up/login with GitHub
3. Click "Import Project"
4. Select your chemnet-frontend repository
5. Set environment variables:
   - `NEXT_PUBLIC_API_URL=https://your-railway-app-url.railway.app`
6. Deploy!

## Step 4: Update CORS Settings

After deployment, update your Django settings.py:
```python
CORS_ALLOWED_ORIGINS = [
    "https://your-vercel-app.vercel.app",  # Your actual Vercel URL
    "http://localhost:3000",  # Keep for local development
]
```

## Step 5: Test Your Live Website!

Your app will be live at: https://your-vercel-app.vercel.app

## Alternative Platforms:

### For Django:
- Heroku (has free tier limitations)
- Render (good free tier)
- PythonAnywhere (you already have this)

### For Next.js:
- Netlify (alternative to Vercel)
- Railway (can host both frontend and backend)

## Custom Domain (Optional):
Both Vercel and Railway support custom domains if you have one.

## Environment Variables Summary:

### Railway (Django):
- `DEBUG=False`
- `SECRET_KEY=your-secret-key`

### Vercel (Next.js):
- `NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app`