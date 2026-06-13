# Deployment Guide

## Render

1. Create a new Web Service on Render and connect this repository.
2. Set the build command to:
   - pip install -r requirements.txt
3. Set the start command to:
   - gunicorn app:app --bind 0.0.0.0:$PORT
4. Add environment variables from .env.example:
   - FLASK_ENV=production
   - SECRET_KEY=<strong value>
   - DATABASE_URL=sqlite:///retina_heart_predict.db
   - PORT=10000 (Render will provide this automatically)
5. Deploy the service and visit the generated URL.

## Railway

1. Create a new project on Railway and import this repository.
2. Use the existing Python service template or add a Dockerfile.
3. Set the startup command to:
   - gunicorn app:app --bind 0.0.0.0:$PORT
4. Add the same environment variables listed in .env.example.
5. Deploy and confirm the health endpoint loads.

## Notes

- The app uses SQLite by default for local development.
- For production, consider migrating to PostgreSQL by setting DATABASE_URL to a hosted PostgreSQL service.
- Keep SECRET_KEY unique and private.
