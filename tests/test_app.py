from app import app, DB, bcrypt
from app.models import User
from app.ml.evaluation import generate_evaluation_artifacts


def _login(client, email='history@example.com', password='strongpass123'):
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if user is None:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username='historyuser', email=email, password=hashed_password)
            DB.session.add(user)
            DB.session.commit()

    return client.post('/login', data={'email': email, 'password': password}, follow_redirects=True)


def test_history_page_exposes_search_and_pagination_controls():
    with app.app_context():
        DB.session.query(User).filter(User.email == 'history@example.com').delete()
        DB.session.commit()

    client = app.test_client()
    response = _login(client)

    assert response.status_code == 200

    history_response = client.get('/history')
    assert history_response.status_code == 200
    assert b'Search predictions' in history_response.data
    assert b'Filter by result' in history_response.data
    assert b'Previous' in history_response.data or b'Next' in history_response.data


def test_history_detail_route_returns_not_found_for_unknown_record():
    client = app.test_client()
    _login(client)

    response = client.get('/history/999999')

    assert response.status_code == 404


def test_auth_pages_render_readable_labels():
    client = app.test_client()

    login_response = client.get('/login')
    register_response = client.get('/register')

    assert login_response.status_code == 200
    assert b'Email' in login_response.data
    assert b'Password' in login_response.data
    assert b'Login' in login_response.data

    assert register_response.status_code == 200
    assert b'Username' in register_response.data
    assert b'Email' in register_response.data
    assert b'Password' in register_response.data
    assert b'Register' in register_response.data


def test_contact_information_is_updated_on_public_pages():
    client = app.test_client()

    home_response = client.get('/')

    assert home_response.status_code == 200
    assert b'Hyderabad, Telangana, India' in home_response.data
    assert b'csd23750026@matrusri.edu.in' in home_response.data


def test_admin_dashboard_renders_stats_and_charts():
    with app.app_context():
        DB.session.query(User).filter(User.email == 'admin@example.com').delete()
        DB.session.commit()
        admin = User(
            username='adminuser',
            email='admin@example.com',
            password=bcrypt.generate_password_hash('adminpass123').decode('utf-8'),
            role='admin',
        )
        DB.session.add(admin)
        DB.session.commit()

    client = app.test_client()
    response = client.post('/login', data={'email': 'admin@example.com', 'password': 'adminpass123'}, follow_redirects=True)

    assert response.status_code == 200

    dashboard_response = client.get('/admin')
    assert dashboard_response.status_code == 200
    assert b'Total Users' in dashboard_response.data
    assert b'Total Predictions' in dashboard_response.data
    assert b'Model Accuracy' in dashboard_response.data
    assert b'Prediction Statistics' in dashboard_response.data
    assert b'Chart.js' in dashboard_response.data


def test_evaluation_artifacts_generate_metrics_and_plots(tmp_path):
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_prob = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2]]

    metrics, paths = generate_evaluation_artifacts(y_true, y_pred, y_prob, output_dir=tmp_path)

    assert set(['accuracy', 'precision', 'recall', 'f1_score']).issubset(metrics)
    assert metrics['accuracy'] >= 0
    assert metrics['precision'] >= 0
    assert metrics['recall'] >= 0
    assert metrics['f1_score'] >= 0
    assert all(__import__('pathlib').Path(path).exists() for path in paths.values())
