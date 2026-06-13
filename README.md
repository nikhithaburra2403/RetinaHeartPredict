# RetinaHeartPredict

RetinaHeartPredict is a Flask-based web application that combines retinal image analysis with machine learning to predict cardiovascular risk indicators.

## Production Deployment Setup

### Environment variables

Copy .env.example to .env and update the values before deploying.

### Local run

1. python -m pip install -r requirements.txt
2. python app.py

### Production run with Gunicorn

gunicorn app:app --bind 0.0.0.0:$PORT

## Docker

Build:

docker build -t retinaheartpredict .

Run:

docker run -p 5000:5000 --env-file .env retinaheartpredict

## Project Structure

- app/ - Main Flask application package
  - __init__.py - Creates the Flask app, configures SQLite, and registers routes
  - models.py - Database models for users, predictions, and admin settings
  - routes.py - Route handlers for authentication, upload, history, and admin pages
  - ml/ - Machine learning and preprocessing utilities
  - static/ - CSS, JS, and uploaded assets
  - templates/ - HTML pages for the frontend UI
- tests/ - Test cases for routes, models, and ML helpers
- docs/ - Project notes and deployment guidance
- requirements.txt - Python dependencies for the app

## Deployment Targets

- Render: see DEPLOYMENT.md
- Railway: see DEPLOYMENT.md

## Main Features

1. User authentication
2. Retinal image upload
3. Image preprocessing
4. CNN feature extraction
5. Logistic Regression classification
6. Prediction result page
7. Prediction history
8. Admin dashboard
9. Responsive Bootstrap-based UI
