import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from model import EnhancedAttributeClassifier  # Import your model class

# Initialize Flask app
app = Flask(__name__)

# Load model
num_attributes = 77
num_categories = 50  # Replace with your actual number of categories
device = torch.device("cpu")
model = EnhancedAttributeClassifier(num_attributes, num_categories)
model.load_state_dict(torch.load('models/enhanced_attribute_classifier.pth', map_location=device))
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request
        data = request.get_json()
        image = data['image']  # Base64-encoded image
        category = data['category']  # Category index

        # Decode and preprocess image
        image = preprocess(image).unsqueeze(0).to(device)

        # Convert category to tensor
        category = torch.tensor([category]).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(image, category).cpu().numpy()

        return jsonify({'predictions': outputs.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
