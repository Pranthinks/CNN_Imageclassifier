import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import torchvision.transforms as transforms

# ---------------------
# Define the ExtendedCNN architecture
# ---------------------
class ExtendedCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ExtendedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm2d(32)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjust the flattened size based on input dimensions (224x224 in this case)
        self.flattened_size = 32 * 3 * 3
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------------------
# Global settings and model loading
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image normalization parameters and transformations
mean = np.array([0.5728, 0.4417, 0.3750])
std = np.array([0.2843, 0.2449, 0.2330])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Class names corresponding to the model's output classes
classes = ['Allu Arjun', 'Mahesh Babu', 'NTR', 'Prabhas']

# Load the saved model (ensure the .pth file is in the same directory or update the path)
model = ExtendedCNN(num_classes=4)
model.load_state_dict(torch.load("extended_cnn.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# ---------------------
# Flask application setup
# ---------------------
app = Flask(__name__, static_folder="../frontend/public")
CORS(app)

# Serve static images from the public folder
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory("../frontend/public/images", filename)

# Endpoint for file upload, face detection, and classification
@app.route('/api/predict', methods=['POST'])
def predict():
    # Debug: print incoming request files
    print("Request files keys:", list(request.files.keys()))
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    print("Received file:", file.filename)
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load the image (RGB format) using face_recognition
        image = face_recognition.load_image_file(file)
    except Exception as e:
        return jsonify({'error': f'Error loading image: {str(e)}'}), 400

    # Detect faces in the image
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print('NO face detected!!!')
        return jsonify({'error': 'No face detected in the image'}), 400

    # Process the first detected face
    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]
    # Resize face to 224x224 pixels
    face_resized = cv2.resize(face, (224, 224))
    
    # Convert the numpy array to a PIL image for the transformation pipeline
    face_pil = Image.fromarray(face_resized)
    
    # Apply the defined transformations
    input_tensor = test_transforms(face_pil).unsqueeze(0).to(device)

    # Perform classification using the loaded model
    with torch.no_grad():
        output = model(input_tensor)
        # Compute softmax probabilities for percentage confidence
        probabilities = torch.softmax(output, dim=1)
        percentages = probabilities * 100

        # Identify the predicted class
        predicted_idx = torch.argmax(probabilities, dim=1)
        predicted_class = classes[predicted_idx.item()]
        predicted_percentage = percentages[0][predicted_idx].item()
        
        # Create a dictionary mapping each class to its percentage confidence
        percentage_dict = {classes[i]: round(percentages[0][i].item(), 2) for i in range(len(classes))}

    # Print detailed classification results in the backend terminal
    print("Detailed Confidence Percentages:")
    print(percentage_dict)
    print(f"Predicted: {predicted_class} with {predicted_percentage:.2f}% confidence")
    
    # Return only minimal information to the frontend
    return jsonify({
        'prediction': predicted_class,
        'message': 'Image processed and classified successfully.'
    })

if __name__ == '__main__':
    app.run(debug=True)
