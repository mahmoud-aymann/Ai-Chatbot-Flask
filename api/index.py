from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Simple fallback responses
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

# Predefined responses for common questions
responses = {
    'greeting': [
        "Hello! I'm Mahmoud Ayman's AI assistant. How can I help you today?",
        "Hi there! I'm here to tell you about Mahmoud Ayman's background and experience.",
        "Greetings! I'm an AI chatbot representing Mahmoud Ayman. What would you like to know?"
    ],
    'about': [
        "I'm Mahmoud Ayman, a Communications and Electronics Engineering student pursuing an advanced AI & Machine Learning diploma. I'm passionate about integrating electronics with AI, especially in computer vision applications.",
        "I'm an AI and engineering student with a strong focus on machine learning, computer vision, and natural language processing. I love working on innovative projects that combine hardware and software.",
        "I'm a dedicated student in the field of AI and machine learning, with hands-on experience in various technologies and a passion for creating impactful solutions."
    ],
    'skills': [
        "My key skills include: Artificial Intelligence & Machine Learning (NLP, Computer Vision, LLMs, Generative AI), Python programming, Front-end Development (HTML5, CSS, JavaScript), and Data Science.",
        "I specialize in AI technologies including PyTorch, scikit-learn, computer vision, natural language processing, and web development. I also have experience with client-based project delivery.",
        "My technical expertise covers machine learning, deep learning, computer vision, natural language processing, Python development, and full-stack web development."
    ],
    'projects': [
        "I've worked on 100+ client projects including 15+ front-end websites for Saudi clients and multiple AI-driven solutions. I've successfully delivered a wide range of projects in AI, data science, and web development.",
        "My project portfolio includes various AI applications, web development projects, and data science solutions. I've collaborated with international partners and delivered projects for clients worldwide.",
        "I've completed numerous projects in AI, machine learning, and web development, working with clients from different countries and delivering innovative solutions."
    ],
    'education': [
        "I'm currently pursuing an advanced AI & Machine Learning diploma, covering NLP, Computer Vision, LLMs, and Generative AI. I'm also studying Communications and Electronics Engineering.",
        "My educational background includes studies in Communications and Electronics Engineering with a specialization in AI and Machine Learning. I'm continuously learning about the latest technologies in the field.",
        "I'm enrolled in an AI & Machine Learning diploma program that covers comprehensive topics including natural language processing, computer vision, and generative AI technologies."
    ],
    'experience': [
        "I have solid professional experience as a freelance developer and entrepreneur. I co-manage a small business with an international partner, delivering AI, data science, and web development services.",
        "My professional experience includes freelance development work and entrepreneurship. I've worked with 100+ clients and successfully delivered various AI and web development projects.",
        "I bring both technical expertise and business experience to my work, having managed projects and worked with clients internationally in the AI and technology space."
    ]
}

def get_response(user_input):
    """Get appropriate response based on user input"""
    user_lower = user_input.lower()
    
    # Check for greeting patterns
    if any(word in user_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return responses['greeting'][0]
    
    # Check for about patterns
    elif any(word in user_lower for word in ['who are you', 'tell me about yourself', 'about you', 'introduce yourself']):
        return responses['about'][0]
    
    # Check for skills patterns
    elif any(word in user_lower for word in ['skills', 'what can you do', 'abilities', 'expertise', 'technologies']):
        return responses['skills'][0]
    
    # Check for projects patterns
    elif any(word in user_lower for word in ['projects', 'work', 'portfolio', 'experience', 'what have you done']):
        return responses['projects'][0]
    
    # Check for education patterns
    elif any(word in user_lower for word in ['education', 'studies', 'degree', 'university', 'college', 'learning']):
        return responses['education'][0]
    
    # Check for experience patterns
    elif any(word in user_lower for word in ['experience', 'professional', 'career', 'job', 'work history']):
        return responses['experience'][0]
    
    # Default fallback
    else:
        return get_fallback_response(user_input)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chatbot - Mahmoud Ayman</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
            }
            .profile-img { 
                width: 120px; 
                height: 120px; 
                border-radius: 50%; 
                margin-bottom: 20px;
                border: 4px solid #007bff;
                box-shadow: 0 8px 20px rgba(0,123,255,0.3);
            }
            h1 { 
                color: #333; 
                margin-bottom: 10px; 
                font-size: 2.5em;
            }
            .subtitle { 
                color: #666; 
                margin-bottom: 30px; 
                font-size: 1.2em;
            }
            .chat-container { 
                border: 2px solid #e9ecef; 
                border-radius: 15px; 
                height: 400px; 
                overflow-y: auto; 
                padding: 20px; 
                margin-bottom: 20px; 
                background: #f8f9fa;
            }
            .input-container { 
                display: flex; 
                gap: 10px; 
            }
            input[type="text"] { 
                flex: 1; 
                padding: 15px; 
                border: 2px solid #ddd; 
                border-radius: 25px; 
                font-size: 16px; 
                outline: none;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus {
                border-color: #007bff;
            }
            button { 
                padding: 15px 30px; 
                background: linear-gradient(135deg, #007bff, #0056b3); 
                color: white; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,123,255,0.4);
            }
            .message { 
                margin-bottom: 15px; 
                padding: 15px; 
                border-radius: 15px; 
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message { 
                background: linear-gradient(135deg, #007bff, #0056b3); 
                color: white; 
                margin-left: 20%; 
                text-align: right;
            }
            .bot-message { 
                background: linear-gradient(135deg, #e9ecef, #f8f9fa); 
                color: #333; 
                margin-right: 20%; 
                border-left: 4px solid #007bff;
            }
            .welcome-message {
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                text-align: center;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Chatbot</h1>
                <p class="subtitle">Meet Mahmoud Ayman - AI & Engineering Student</p>
            </div>
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message welcome-message">
                    Hello! I'm Mahmoud Ayman's AI assistant. Ask me about my studies, experience, skills, or projects!
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
    try:
        value = request.form.get('message', '')
        print(f"Received message: '{value}'")
        
        if not value:
            return jsonify({'error': 'Please enter a message.'}), 400
        
        # Get response using simple pattern matching
        response = get_response(value)
        
        return jsonify({
            'response': response, 
            'intent': 'simple', 
            'confidence': 1.0
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Load model when the module is imported
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
else:
    # This runs when deployed
    pass