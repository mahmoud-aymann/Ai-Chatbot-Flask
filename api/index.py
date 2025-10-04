from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle, json
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import os

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

# Global variables for model components
vectorizer = None
le = None
model = None
data = None

def load_model():
    global vectorizer, le, model, data
    try:
        # Try to load files from different possible locations
        base_paths = ['.', '/var/task', '/tmp']
        
        for base_path in base_paths:
            try:
                print(f"Trying to load from {base_path}...")
                
                # Load vectorizer
                vectorizer_path = os.path.join(base_path, 'vectorizer.pkl')
                if os.path.exists(vectorizer_path):
                    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
                    print(f"Vectorizer loaded from {vectorizer_path}")
                    break
            except Exception as e:
                print(f"Failed to load from {base_path}: {e}")
                continue
        
        if vectorizer is None:
            print("Could not load vectorizer from any path")
            return False
            
        print(f"Vectorizer loaded. Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Load label encoder
        for base_path in base_paths:
            try:
                le_path = os.path.join(base_path, 'label_encoder.pkl')
                if os.path.exists(le_path):
                    le = pickle.load(open(le_path, 'rb'))
                    print(f"Label encoder loaded from {le_path}")
                    break
            except Exception as e:
                print(f"Failed to load label encoder from {base_path}: {e}")
                continue
        
        if le is None:
            print("Could not load label encoder from any path")
            return False
            
        print(f"Label encoder loaded. Classes: {len(le.classes_)}")
        
        # Create model
        print("Creating model...")
        model = ChatbotModel(input_dim=len(vectorizer.vocabulary_), hidden_dim=128, output_dim=len(le.classes_))
        
        # Load model weights
        for base_path in base_paths:
            try:
                model_path = os.path.join(base_path, 'model.pth')
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                    print(f"Model weights loaded from {model_path}")
                    break
            except Exception as e:
                print(f"Failed to load model from {base_path}: {e}")
                continue
        
        if model is None:
            print("Could not load model from any path")
            return False
        
        # Load data
        for base_path in base_paths:
            try:
                data_path = os.path.join(base_path, 'data.json')
                if os.path.exists(data_path):
                    with open(data_path, encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"Data loaded from {data_path}")
                    break
            except Exception as e:
                print(f"Failed to load data from {base_path}: {e}")
                continue
        
        if data is None:
            print("Could not load data from any path")
            return False
            
        print(f"Data loaded. Number of intents: {len(data)}")
        print("Model and data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model or data: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    try:
        return render_template('home.html')
    except Exception as e:
        print(f"Error loading template: {e}")
        # Fallback HTML if template fails
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Chatbot - Mahmoud Ayman</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .header { text-align: center; margin-bottom: 30px; }
                .profile-img { width: 120px; height: 120px; border-radius: 50%; margin-bottom: 20px; }
                h1 { color: #333; margin-bottom: 10px; }
                .subtitle { color: #666; margin-bottom: 30px; }
                .chat-container { border: 1px solid #ddd; border-radius: 10px; height: 400px; overflow-y: auto; padding: 20px; margin-bottom: 20px; background: #fafafa; }
                .input-container { display: flex; gap: 10px; }
                input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
                button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
                button:hover { background: #0056b3; }
                .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
                .user-message { background: #007bff; color: white; margin-left: 20%; }
                .bot-message { background: #e9ecef; color: #333; margin-right: 20%; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 AI Chatbot</h1>
                    <p class="subtitle">Meet Mahmoud Ayman - AI & Engineering Student</p>
                </div>
                <div class="chat-container" id="chatContainer">
                    <div class="message bot-message">
                        Hello! I'm Mahmoud Ayman's AI assistant. Ask me about my studies, experience, or projects!
                    </div>
                </div>
                <div class="input-container">
                    <input type="text" id="messageInput" placeholder="Type your message here..." onkeypress="handleKeyPress(event)">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
            <script>
                function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    if (message) {
                        addMessage(message, true);
                        input.value = '';
                        
                        // Send to backend
                        fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                            body: 'message=' + encodeURIComponent(message)
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                addMessage('Error: ' + data.error, false);
                            } else {
                                addMessage(data.response, false);
                            }
                        })
                        .catch(error => {
                            addMessage('Sorry, I encountered an error. Please try again.', false);
                        });
                    }
                }
                
                function addMessage(message, isUser = false) {
                    const chatContainer = document.getElementById('chatContainer');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
                    messageDiv.textContent = message;
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                function handleKeyPress(event) {
                    if (event.key === 'Enter') {
                        sendMessage();
                    }
                }
            </script>
        </body>
        </html>
        '''

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

# Load model when the module is imported
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    # Load model when deployed
    load_model()
