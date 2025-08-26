# Celebrity Image Classifier using Custom CNN Architecture

## Overview

This project implements a high-performance celebrity image classification system using a custom-designed Convolutional Neural Network (CNN) architecture built with PyTorch. The model achieves **90% accuracy** in classifying images of four popular Telugu cinema actors and features a complete full-stack implementation with a React frontend and Flask backend.

## Project Highlights

- **Custom CNN Architecture**: Designed from scratch with 6 convolutional layers and advanced regularization techniques
- **High Accuracy**: Achieved 90% classification accuracy on test dataset
- **Real-time Face Detection**: Integrated face detection using face_recognition library
- **Full-stack Implementation**: Complete web application with React frontend and Flask API backend
- **Production Ready**: Optimized model with proper preprocessing pipeline and error handling

## Technical Architecture

### Model Architecture Details

Our custom **ExtendedCNN** architecture consists of:

**Convolutional Layers:**
- **6 Convolutional Blocks** with progressive feature extraction
- **Batch Normalization** after each convolution for training stability
- **MaxPooling** layers for spatial dimension reduction
- **ReLU Activation** functions for non-linearity

**Architecture Breakdown:**
```
Input: 224×224×3 RGB Images
├── Conv1: 3→32 channels, 3×3 kernel + BatchNorm + MaxPool
├── Conv2: 32→64 channels, 5×5 kernel + BatchNorm + MaxPool
├── Conv3: 64→48 channels, 3×3 kernel + BatchNorm + MaxPool
├── Conv4: 48→128 channels, 5×5 kernel + BatchNorm + MaxPool
├── Conv5: 128→64 channels, 3×3 kernel + BatchNorm + MaxPool
├── Conv6: 64→32 channels, 5×5 kernel + BatchNorm + MaxPool
├── Flatten: 32×3×3 = 288 features
├── FC1: 288→1024 neurons + ReLU + Dropout(0.8)
└── FC2: 1024→4 classes (Output)
```

**Key Technical Features:**
- **Varying Kernel Sizes**: Mix of 3×3 and 5×5 kernels for multi-scale feature extraction
- **Progressive Channel Reduction**: Strategic channel management for computational efficiency
- **High Dropout Rate (0.8)**: Aggressive regularization to prevent overfitting
- **Batch Normalization**: Accelerated training and improved gradient flow

### Data Preprocessing Pipeline

**Image Normalization:**
```python
mean = [0.5728, 0.4417, 0.3750]  # Dataset-specific RGB means
std = [0.2843, 0.2449, 0.2330]   # Dataset-specific RGB standard deviations
```

**Preprocessing Steps:**
1. **Face Detection**: Automatic face localization using `face_recognition` library
2. **Face Extraction**: Crop detected face regions for focused classification
3. **Resize**: Standardize all images to 224×224 pixels
4. **Normalization**: Apply dataset-specific mean and standard deviation
5. **Tensor Conversion**: Transform to PyTorch tensors for model input

## Classification Classes

The model classifies between four renowned Telugu cinema actors:

1. **Allu Arjun** - Stylish Star of Tollywood
2. **Mahesh Babu** - Superstar Prince
3. **NTR (N. T. Rama Rao Jr.)** - Young Tiger
4. **Prabhas** - Rebel Star & Pan-India Icon

## Technology Stack

### Backend (Flask API)
- **PyTorch**: Deep learning framework for model implementation
- **OpenCV**: Computer vision operations and image processing
- **PIL (Pillow)**: Image handling and transformations
- **face_recognition**: Face detection and localization
- **Flask**: REST API development
- **Flask-CORS**: Cross-origin resource sharing

### Frontend (React)
- **React**: Component-based UI development
- **Axios**: HTTP client for API communication
- **CSS3**: Modern styling and animations
- **File Upload**: Drag-and-drop image upload interface

### Machine Learning Pipeline
- **Custom CNN Architecture**: Hand-designed deep learning model
- **Batch Normalization**: Training stabilization technique
- **Dropout Regularization**: Overfitting prevention
- **Adam Optimizer**: Adaptive learning rate optimization
- **Cross-Entropy Loss**: Multi-class classification loss function

## Project Structure

```
celebrity-classifier/
├── backend/
│   ├── server.py              # Flask API server
│   ├── extended_cnn.pth       # Trained model weights
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styling
│   │   └── index.js          # React entry point
│   ├── public/
│   │   └── images/           # Sample celebrity images
│   └── package.json          # Node.js dependencies
└── README.md                 # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- CUDA-compatible GPU (optional, for faster inference)

### Backend Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/celebrity-classifier.git
cd celebrity-classifier/backend
```

2. **Install Python dependencies:**
```bash
pip install torch torchvision flask flask-cors opencv-python pillow face-recognition numpy
```

3. **Start the Flask server:**
```bash
python server.py
```
Server will run on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd ../frontend
```

2. **Install Node.js dependencies:**
```bash
npm install
```

3. **Start the React development server:**
```bash
npm start
```
Application will open on `http://localhost:3000`

## Usage Instructions

1. **Access the Web Application**: Open your browser to `http://localhost:3000`

2. **Upload Image**: Click "Choose a File" and select a celebrity image

3. **Get Prediction**: Click "Upload" to classify the image

4. **View Results**: The model will display the predicted celebrity name

## Model Performance

**Training Metrics:**
- **Accuracy**: 90% on test dataset
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with adaptive learning rate
- **Regularization**: Batch Normalization + Dropout (0.8)
- **Training Time**: ~2-3 hours on modern GPU

**Performance Optimizations:**
- **Face Detection**: Focuses classification on facial features only
- **Batch Processing**: Efficient tensor operations
- **Model Compression**: Optimized architecture for fast inference
- **Error Handling**: Graceful handling of edge cases

## Future Enhancements

**Model Improvements:**
- [ ] Expand to more celebrity classes
- [ ] Implement attention mechanisms
- [ ] Add confidence score visualization
- [ ] Multi-face detection and classification

**Technical Upgrades:**
- [ ] Model quantization for mobile deployment
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)
- [ ] Real-time webcam classification
- [ ] Progressive Web App (PWA) conversion

**UI/UX Enhancements:**
- [ ] Drag-and-drop file upload
- [ ] Image crop/edit functionality  
- [ ] Celebrity information display
- [ ] Classification confidence visualization

## Model Architecture Visualization

```
Input Image (224×224×3)
        ↓
   Face Detection
        ↓
   Face Extraction  
        ↓
    Preprocessing
        ↓
┌─────────────────────┐
│   Conv Block 1      │ → 32 features
│   Conv Block 2      │ → 64 features  
│   Conv Block 3      │ → 48 features
│   Conv Block 4      │ → 128 features
│   Conv Block 5      │ → 64 features
│   Conv Block 6      │ → 32 features
└─────────────────────┘
        ↓
    Fully Connected
        ↓
   Classification
        ↓
┌─────────────────────┐
│   Allu Arjun        │
│   Mahesh Babu       │  
│   NTR               │
│   Prabhas           │
└─────────────────────┘
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

**Areas for Contribution:**
- Model architecture improvements
- Additional celebrity classes  
- UI/UX enhancements
- Performance optimizations
- Documentation improvements

**Star this repository if you found it helpful!**
