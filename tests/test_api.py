from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "boston-house-price-xgboost"
    assert data["version"] == "1"


def test_prediction_endpoint():

    payload = {
        "crime_rate": 0.00632,
        "residential_land_pct": 18.0,
        "industrial_land_pct": 2.31,
        "river_boundary": 0,
        "nitric_oxide_concentration": 0.538,
        "avg_rooms": 6.575,
        "old_housing_pct": 65.2,
        "employment_distance": 4.09,
        "highway_access_index": 1,
        "property_tax_rate": 296.0,
        "student_teacher_ratio": 15.3,
        "demographic_index": 396.9,
        "lower_status_pct": 4.98,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], float)

    assert data["model"] == "boston-house-price-xgboost"
    assert data["model_version"] == "1"


def test_prediction_rejects_missing_feature():

    payload = {
        "crime_rate": 0.00632,
        "residential_land_pct": 18.0,
        "industrial_land_pct": 2.31,
        "river_boundary": 0,
        "nitric_oxide_concentration": 0.538,
        "avg_rooms": 6.575,
        "old_housing_pct": 65.2,
        "employment_distance": 4.09,
        "highway_access_index": 1,
        "property_tax_rate": 296.0,
        "student_teacher_ratio": 15.3,
        "demographic_index": 396.9
        # lower_status_pct intentionally missing
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422