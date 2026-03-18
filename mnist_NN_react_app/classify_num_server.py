import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Maximal permissive CORS für Development
CORS(app,
     origins="*",
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

MODEL_PATH = Path(__file__).resolve().parent.parent / 'mnist_neural_net' / 'models' / 'mnist_model_v4_94_57_batched.npz'

def load_model():
    print("Loading model...")

    data = np.load(MODEL_PATH)
    print("Model data keys:", list(data.keys()))
    
    # Finde heraus, wie viele Layer wir haben
    num_layers = len([key for key in data.keys() if key.startswith('weights_')])
    print(f"Number of layers: {num_layers}")
    
    w = [data[f'weights_{i}'] for i in range(num_layers)]
    b = [data[f'biases_{i}'] for i in range(num_layers)]
    return w, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    output = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True) # axis=1 means sum each row, keepdims=True ensure the result is (n, 1) and not (n,)
    return output


def feed_forward_neural_network(neuron_vals, w, b):
    activations = [neuron_vals]

    for layer_weights, layer_biases in zip(w, b):
        neuron_vals_before_activation = neuron_vals @ layer_weights + layer_biases # first matrix multiplication, then + bias

        if layer_weights is not w[-1]:
            # we only want to apply the activation function to the hidden layers, not the output layer
            neuron_vals = sigmoid(neuron_vals_before_activation)
        else:
            neuron_vals = neuron_vals_before_activation  # output layer without activation function
            
        activations.append(neuron_vals)
        
    return activations

@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        data = request.json
        print("data:", data)
        pixels = np.array(data['pixels'], dtype=np.float32).reshape(1, -1)  # Reshape für dein Modell
        print(f"Pixels shape: {pixels.shape}")
        
        # Normalisierung (falls noch nicht gemacht)
        if np.max(pixels) > 1:
            pixels = pixels / 255.0

        w, b = load_model()

        print("Model weights and biases loaded.")
        print(w, b)

        # Feed Forward durch dein Netzwerk
        activations = feed_forward_neural_network(pixels, w, b)
        output = activations[-1]  # Letzte Schicht
        
        # Softmax auf Output anwenden
        probabilities = softmax(output)[0]  # [0] um 2D zu 1D zu machen
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        response = jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        })
        # Explizite CORS-Headers
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        error_response = jsonify({'error': str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

@app.route('/health', methods=['GET'])
def health():
    response = jsonify({'status': 'Server is running!'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    print("Starting Python Classification Server...")
    print("Server will run on http://localhost:5000")
    app.run(debug=True, port=5001, host='0.0.0.0')