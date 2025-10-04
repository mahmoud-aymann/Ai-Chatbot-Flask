from setuptools import setup, find_packages

setup(
    name="ai-chatbot-flask",
    version="1.0.0",
    description="AI Chatbot for Mahmoud Ayman",
    py_modules=["deploy_model"],
    install_requires=[
        "Flask==2.3.3",
        "torch==2.2.0",
        "numpy==1.26.0",
        "scikit-learn==1.3.0",
        "gunicorn==21.2.0"
    ],
    python_requires=">=3.11",
)
