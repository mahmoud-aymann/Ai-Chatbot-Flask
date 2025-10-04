# AI Chatbot Deployment

This is a Flask web application that deploys a PyTorch-based chatbot model.

## Files Structure

- `deploy_model.py` - Main Flask application
- `model.pth` - Trained PyTorch model
- `vectorizer.pkl` - TF-IDF vectorizer for text preprocessing
- `label_encoder.pkl` - Label encoder for intent classification
- `data.json` - Training data with intents and responses
- `templates/home.html` - Web interface template
- `static/style.css` - CSS styles for the web interface
- `static/script.js` - JavaScript functionality for the chat
- `requirements.txt` - Python dependencies

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python deploy_model.py
```

2. Open your web browser and go to:
```
http://localhost:5000
```

## API Endpoints

### Web Interface
- `GET /` - Main chat interface

### API Endpoints
- `POST /predict` - Form-based prediction endpoint
- `POST /api/predict` - JSON-based prediction endpoint

### Example API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

## Features

- Modern, responsive web interface
- Real-time chat with the AI model
- Intent classification and response generation
- Error handling and loading states
- Both web and API interfaces

## Model Architecture

The chatbot uses a simple neural network with:
- Input layer: 2000 features (TF-IDF vectors)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: Softmax for intent classification

## Troubleshooting

If you encounter issues:
1. Make sure all model files are present
2. Check that all dependencies are installed
3. Verify the model files are not corrupted
4. Check the console output for error messages
