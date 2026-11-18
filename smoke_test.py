
from app import app, DISEASES


client = app.test_client()

EXPECTED_DISEASES = {
    'heart',
    'diabetes',
    'kidney',
    'breast_cancer'
}


# Check disease configuration.
assert set(DISEASES.keys()) == EXPECTED_DISEASES

# Check home route.
response = client.get('/')
assert response.status_code == 200
print('Home route: OK')

# Check health route.
response = client.get('/health')
assert response.status_code == 200

health_data = response.get_json()
assert health_data['status'] == 'ok'
print('Health route: OK')


# Check all four models and prediction endpoints.
for disease, schema in DISEASES.items():

    # Model availability.
    assert health_data['models'][disease] is True, (
        f'{disease} model is missing'
    )

    # Schema endpoint.
    response = client.get(f'/api/schema/{disease}')
    assert response.status_code == 200, (
        f'{disease} schema failed'
    )

    # Generate inputs using schema defaults.
    payload = {
        field[0]: field[5]
        for field in schema['fields']
    }

    # Prediction endpoint.
    response = client.post(
        f'/api/predict/{disease}',
        json=payload
    )

    assert response.status_code == 200, (
        f'{disease} prediction failed: {response.get_json()}'
    )

    result = response.get_json()

    assert 'prediction' in result
    assert 'label' in result
    assert 'model_score' in result

    print(f'{disease}: schema OK, prediction OK')


# Check invalid disease.
response = client.get('/api/schema/parkinsons')
assert response.status_code == 404

print('\nAll four disease models passed the smoke tests!')
