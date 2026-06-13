from app import DB, login_manager
from flask_login import UserMixin
from datetime import datetime


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(DB.Model, UserMixin):
    id = DB.Column(DB.Integer, primary_key=True)
    username = DB.Column(DB.String(80), unique=True, nullable=False)
    email = DB.Column(DB.String(120), unique=True, nullable=False)
    password = DB.Column(DB.String(200), nullable=False)
    role = DB.Column(DB.String(30), default='user')


class Prediction(DB.Model):
    id = DB.Column(DB.Integer, primary_key=True)
    user_id = DB.Column(DB.Integer, DB.ForeignKey('user.id'), nullable=False)
    image_path = DB.Column(DB.String(255), nullable=False)
    result = DB.Column(DB.String(100), nullable=False)
    confidence = DB.Column(DB.Float, default=0.0)
    created_at = DB.Column(DB.DateTime, default=datetime.utcnow)
