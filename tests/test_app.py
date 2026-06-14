import io

import numpy as np

from app import app, DB, bcrypt
from app.models import User
from app.ml.evaluation import generate_evaluation_artifacts
from app.ml.hybrid_model import HybridRetinaModel


def _login(client, email='history@example.com', password='strongpass123'):
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if user is None:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(username='historyuser', email=email, password=hashed_password)
            DB.session.add(user)
            DB.session.commit()

    return client.post('/login', data={'email': email, 'password': password}, follow_redirects=True)


def test_login_migrates_legacy_plain_text_passwords_to_bcrypt():
    with app.app_context():
        DB.session.query(User).filter(User.email == 'legacy@example.com').delete()
        DB.session.commit()

        legacy_user = User(
            username='legacyuser',
            email='legacy@example.com',
            password='plain-text-password',
        )
        DB.session.add(legacy_user)
        DB.session.commit()

    client = app.test_client()
    response = client.post(
        '/login',
        data={'email': 'legacy@example.com', 'password': 'plain-text-password'},
        follow_redirects=True,
    )

    assert response.status_code == 200

    with app.app_context():
        migrated_user = User.query.filter_by(email='legacy@example.com').first()
        assert migrated_user is not None
        assert migrated_user.password != 'plain-text-password'
        assert bcrypt.check_password_hash(migrated_user.password, 'plain-text-password') is True


def test_upload_form_posts_to_predict_route():
    client = app.test_client()
    _login(client)

    response = client.get('/upload')

    assert response.status_code == 200
    assert b'action="/predict"' in response.data
    assert b'name="image"' in response.data


def test_predict_route_renders_result_page_using_prediction_service(monkeypatch):
    calls = {'count': 0}

    class FakePredictionService:
        def predict_from_file(self, image_path):
            calls['count'] += 1
            return {
                'prediction_label': 'At Risk',
                'confidence_score': 0.87,
                'risk_level': 'High',
                'risk_label': 'High Risk',
                'feature_vector_shape': [1, 1],
            }

    monkeypatch.setattr('app.routes.PredictionService', FakePredictionService)

    client = app.test_client()
    _login(client)

    response = client.post(
        '/predict',
        data={'image': (io.BytesIO(b'fake-image-bytes'), 'sample.png')},
        content_type='multipart/form-data',
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert calls['count'] == 1
    assert b'At Risk' in response.data
    assert b'87.0%' in response.data


def test_hybrid_model_extract_features_works_with_keras_model():
    model = HybridRetinaModel(input_shape=(32, 32, 3), num_classes=2)

    batch = np.zeros((1, 32, 32, 3), dtype=np.float32)
    features = model.extract_features(batch)

    assert features.shape[0] == 1
    assert features.shape[1] == 64


def test_prediction_service_falls_back_when_classifier_is_not_fitted(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from app.ml.predictor import PredictionService

    service = PredictionService()
    service.model.extract_features = lambda image_batch: np.ones((1, 64), dtype=np.float32)
    service.model.classifier = SimpleNamespace(
        classes_=[],
        predict=lambda features: np.array([0]),
        predict_proba=lambda features: np.array([[1.0, 0.0]]),
    )

    monkeypatch.setattr('app.ml.predictor.preprocess_retinal_image', lambda image_path, augment=False: np.ones((32, 32, 3), dtype=np.float32))

    result = service.predict_from_file('app/static/uploads/sample.png')

    assert result['prediction_label'] in ('Healthy', 'At Risk')
    assert 0.0 <= result['confidence_score'] <= 1.0


def test_prediction_service_uses_feature_signal_in_fallback(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from app.ml.predictor import PredictionService

    service = PredictionService()

    high_risk_features = np.full((1, 64), 5.0, dtype=np.float32)
    service.model.extract_features = lambda image_batch: high_risk_features
    service.model.classifier = SimpleNamespace(
        classes_=[],
        predict=lambda features: (_ for _ in ()).throw(RuntimeError('not fitted')),
        predict_proba=lambda features: (_ for _ in ()).throw(RuntimeError('not fitted')),
    )

    monkeypatch.setattr('app.ml.predictor.preprocess_retinal_image', lambda image_path, augment=False: np.ones((32, 32, 3), dtype=np.float32))

    result = service.predict_from_file('app/static/uploads/sample.png')

    assert result['prediction_label'] == 'At Risk'
    assert result['risk_level'] in ('Moderate', 'High')
    assert result['confidence_score'] > 0.5


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
