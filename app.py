from flask import Flask, request, jsonify
from plapt import Plapt

app = Flask(__name__)
plapt = Plapt()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'target' not in data or 'smiles' not in data:
        return jsonify({'error': 'Invalid input. Please provide "target" and "smiles" in the request body.'}), 400

    target = data['target']
    smiles = data['smiles']

    try:
        results = plapt.predict_affinity(target, smiles)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)