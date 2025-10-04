from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle, json
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# --- Load PyTorch model ---
class ChatbotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Load the trained model and preprocessing objects
try:
    print("Loading vectorizer...")
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print(f"Vectorizer loaded. Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    print("Loading label encoder...")
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    print(f"Label encoder loaded. Classes: {len(le.classes_)}")
    
    print("Creating model...")
    model = ChatbotModel(input_dim=len(vectorizer.vocabulary_), hidden_dim=128, output_dim=len(le.classes_))
    
    print("Loading model weights...")
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    
    print("Loading data...")
    with open("data.json", encoding='utf-8') as f:
        data = json.load(f)
    print(f"Data loaded. Number of intents: {len(data)}")
    
    # Simple fallback responses without external models
    genai_model = None
    
    print("Model and data loaded successfully!")
    print(f"Model status: {model is not None}")
    print(f"Vectorizer status: {vectorizer is not None}")
    print(f"Label encoder status: {le is not None}")
    print(f"Data status: {data is not None}")
    
except Exception as e:
    print(f"Error loading model or data: {e}")
    import traceback
    traceback.print_exc()
    vectorizer = None
    le = None
    model = None
    data = None
    genai_model = None

def get_fallback_response(user_input):
    """Generate a fallback response using predefined responses"""
    fallback_responses = [
        "That's an interesting question! I'm still learning about that topic. Could you ask me about my studies, experience, or skills instead?",
        "I'm not sure about that specific topic, but I'd love to tell you about my AI and engineering background! What would you like to know?",
        "That's a great question! While I don't have specific information about that, I can share more about my work in AI and computer vision.",
        "I'm still expanding my knowledge in that area. Could you ask me about my projects, education, or professional experience?",
        "That's something I'm still learning about! I'd be happy to discuss my studies in Communications and Electronics Engineering or my AI diploma instead."
    ]
    
    # Simple keyword-based response selection
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ['how', 'what', 'why', 'when', 'where']):
        return fallback_responses[0]
    elif any(word in user_lower for word in ['tell', 'explain', 'describe']):
        return fallback_responses[1]
    elif any(word in user_lower for word in ['can you', 'do you know', 'do you']):
        return fallback_responses[2]
    else:
        return fallback_responses[3]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model components are loaded
    if model is None:
        print("Model is None")
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    if vectorizer is None:
        print("Vectorizer is None")
        return jsonify({'error': 'Vectorizer not loaded properly'}), 500
    
    if le is None:
        print("Label encoder is None")
        return jsonify({'error': 'Label encoder not loaded properly'}), 500
    
    if data is None:
        print("Data is None")
        return jsonify({'error': 'Data not loaded properly'}), 500
    
    value = request.form.get('message', '')
    print(f"Received message: '{value}'")
    
    if not value:
        return jsonify({'error': 'Please enter a message.'}), 400
    
    try:
        # Preprocess input
        print("Preprocessing input...")
        sent = vectorizer.transform([value]).toarray()
        sent_tensor = torch.tensor(sent, dtype=torch.float32)
        print(f"Input shape: {sent_tensor.shape}")
        
        # Get model prediction
        print("Getting model prediction...")
        with torch.no_grad():
            output = model(sent_tensor)
            predicted_index = torch.argmax(output, dim=1).item()
            predicted_label = le.inverse_transform([predicted_index])[0]
            confidence = torch.max(output).item()
        
        print(f"Predicted: {predicted_label}, Confidence: {confidence:.3f}")
        
        # Check confidence threshold
        confidence_threshold = 0.3
        
        if confidence < confidence_threshold:
            # Use fallback response for low confidence predictions
            print("Using fallback response...")
            response = get_fallback_response(value)
            return jsonify({'response': response, 'intent': 'generated', 'confidence': confidence})
        else:
            # Use predefined responses for high confidence predictions
            print("Using predefined response...")
            response = "Sorry, I do not understand..."
            for item in data:
                if item['tag'] == predicted_label:
                    response = np.random.choice(item['responses'])
                    break
            
            return jsonify({'response': response, 'intent': predicted_label, 'confidence': confidence})
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON requests"""
    if model is None or vectorizer is None or le is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        request_data = request.get_json()
        user_input = request_data.get('message', '')
        
        if not user_input:
            return jsonify({'error': 'Please provide a message.'}), 400
        
        # Preprocess input
        input_vector = vectorizer.transform([user_input]).toarray()
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()
            predicted_label = le.inverse_transform([predicted_index])[0]
            confidence = torch.max(output).item()
        
        # Check confidence threshold
        confidence_threshold = 0.3
        
        if confidence < confidence_threshold:
            # Use Generative AI for low confidence predictions
            response = get_fallback_response(user_input)
            return jsonify({'response': response, 'intent': 'generated', 'confidence': confidence})
        else:
            # Use predefined responses for high confidence predictions
            response = "Sorry, I do not understand..."
            for item in data:
                if item['tag'] == predicted_label:
                    response = np.random.choice(item['responses'])
                    break
            
            return jsonify({'response': response, 'intent': predicted_label, 'confidence': confidence})
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
