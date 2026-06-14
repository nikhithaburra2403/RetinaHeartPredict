import os
import re
from werkzeug.utils import secure_filename

from flask import render_template, request, redirect, url_for, flash, jsonify, send_file, abort
from flask_login import login_user, login_required, logout_user, current_user
from app import app, DB, bcrypt
from app.models import User, Prediction
from app.ml.predictor import PredictionService
from app.report_generator import generate_report

UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 5 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


def is_valid_email(email):
    pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
    return re.fullmatch(pattern, email) is not None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not username or len(username) < 3:
            flash('Username must be at least 3 characters long.', 'danger')
            return render_template('register.html')

        if not is_valid_email(email):
            flash('Please enter a valid email address.', 'danger')
            return render_template('register.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('register.html')

        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash('A user with that username or email already exists.', 'danger')
            return render_template('register.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        DB.session.add(user)
        DB.session.commit()

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()

        if user:
            password_ok = False

            try:
                password_ok = bcrypt.check_password_hash(user.password, password)
            except ValueError:
                # Support older records that were stored as plain text before bcrypt migration.
                password_ok = (user.password == password)
                if password_ok:
                    user.password = bcrypt.generate_password_hash(password).decode('utf-8')
                    DB.session.commit()

        if user and password_ok:
            login_user(user)
            flash('Welcome back!', 'success')
            return redirect(url_for('upload'))

        flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file was uploaded.', 'danger')
            return redirect(request.url)

        image_file = request.files['image']
        if image_file.filename == '':
            flash('Please choose a retinal image to upload.', 'danger')
            return redirect(request.url)

        if not allowed_file(image_file.filename):
            flash('Only JPG, JPEG, and PNG files are allowed.', 'danger')
            return redirect(request.url)

        if image_file.content_length and image_file.content_length > MAX_FILE_SIZE:
            flash('File is too large. Maximum size is 5 MB.', 'danger')
            return redirect(request.url)

        filename = secure_filename(image_file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(save_path)

        flash('Image uploaded successfully. Ready for analysis.', 'success')
        return render_template('upload.html', preview_url=url_for('static', filename='uploads/' + filename))

    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    print('[DEBUG] route /predict entered', flush=True)
    if 'image' not in request.files:
        flash('No image file was uploaded.', 'danger')
        return redirect(url_for('upload'))

    image_file = request.files['image']
    if image_file.filename == '':
        flash('Please choose a retinal image to upload.', 'danger')
        return redirect(url_for('upload'))

    if not allowed_file(image_file.filename):
        flash('Only JPG, JPEG, and PNG files are allowed.', 'danger')
        return redirect(url_for('upload'))

    if image_file.content_length and image_file.content_length > MAX_FILE_SIZE:
        flash('File is too large. Maximum size is 5 MB.', 'danger')
        return redirect(url_for('upload'))

    filename = secure_filename(image_file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(save_path)
    print(f'[DEBUG] file saved to {save_path}', flush=True)

    try:
        print('[DEBUG] creating PredictionService', flush=True)
        service = PredictionService()
        print('[DEBUG] model loaded for prediction', flush=True)
        result = service.predict_from_file(save_path)

        print('[DEBUG] prediction completed', flush=True)
        record = Prediction(
            user_id=current_user.id,
            image_path=filename,
            result=result['prediction_label'],
            confidence=result['confidence_score']
        )
        DB.session.add(record)
        DB.session.commit()

        print('[DEBUG] rendering result page', flush=True)
        return render_template(
            'result.html',
            preview_url=url_for('static', filename='uploads/' + filename),
            prediction=result['prediction_label'],
            confidence=float(result['confidence_score']),
            risk_category=result.get('risk_level', 'Moderate'),
            download_report_url=url_for('download_report', prediction_id=record.id),
        )
    except Exception as exc:
        import traceback
        print('[DEBUG] prediction exception', flush=True)
        traceback.print_exc()
        print(f'[DEBUG] exception type={type(exc).__name__} message={exc}', flush=True)
        flash('Prediction failed. Please try another image.', 'danger')
        return redirect(url_for('upload'))


@app.route('/download-report')
@login_required
def download_report():
    prediction_id = request.args.get('prediction_id', type=int)

    if prediction_id is not None:
        record = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first()
        if record is None:
            abort(404)

        patient_id = f'PATIENT-{record.id:04d}'
        prediction_label = record.result
        confidence_score = float(record.confidence or 0.0)
        image_name = os.path.basename(record.image_path)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    else:
        patient_id = request.args.get('patient_id', f'PATIENT-{current_user.id:04d}')
        prediction_label = request.args.get('prediction', 'Healthy / At Risk')
        confidence_score = float(request.args.get('confidence', 0.92))
        image_path = request.args.get('image_path')

        if image_path:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))
        else:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample-retina.png')

    if not os.path.exists(image_path):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample-retina.png')

    recommendations = [
        'Review findings with a clinician for confirmation.',
        'Monitor blood pressure and cardiovascular risk trends.',
        'Schedule a follow-up retinal imaging review.'
    ]

    pdf_path = generate_report(patient_id, prediction_label, confidence_score, recommendations, image_path)
    return send_file(pdf_path, as_attachment=True, download_name=os.path.basename(pdf_path))


@app.route('/result')
@login_required
def result():
    return render_template('result.html', prediction='Healthy / At Risk', confidence=0.92)


@app.route('/history')
@login_required
def history():
    search_term = request.args.get('q', '').strip()
    selected_result = request.args.get('result', 'all')

    query = Prediction.query.filter_by(user_id=current_user.id)

    if search_term:
        term = f'%{search_term}%'
        query = query.filter((Prediction.result.ilike(term)) | (Prediction.image_path.ilike(term)))

    if selected_result in {'Healthy', 'At Risk'}:
        query = query.filter(Prediction.result == selected_result)

    pagination = query.order_by(Prediction.created_at.desc()).paginate(
        page=request.args.get('page', 1, type=int),
        per_page=10,
        error_out=False,
    )

    return render_template(
        'history.html',
        records=pagination.items,
        pagination=pagination,
        query=search_term,
        selected_result=selected_result,
        result_options=['Healthy', 'At Risk'],
    )


@app.route('/history/<int:prediction_id>')
@login_required
def prediction_detail(prediction_id):
    record = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    image_url = url_for('static', filename='uploads/' + os.path.basename(record.image_path))

    return render_template('history_detail.html', record=record, image_url=image_url)


@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Admin access is required.', 'danger')
        return redirect(url_for('home'))

    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    recent_uploads = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
    users = User.query.order_by(User.id.desc()).all()

    confidence_values = [record.confidence or 0.0 for record in Prediction.query.all()]
    avg_confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
    model_accuracy = round(avg_confidence * 100, 2)

    healthy_count = Prediction.query.filter_by(result='Healthy').count()
    at_risk_count = Prediction.query.filter_by(result='At Risk').count()

    return render_template(
        'admin.html',
        users=users,
        predictions=Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all(),
        recent_uploads=recent_uploads,
        total_users=total_users,
        total_predictions=total_predictions,
        model_accuracy=model_accuracy,
        healthy_count=healthy_count,
        at_risk_count=at_risk_count,
        recent_activity=recent_uploads,
    )
