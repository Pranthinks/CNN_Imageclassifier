import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const images = [
    '/images/ntr.jpg',
    '/images/mahesh.jpg',
    '/images/alluarjun.jpg',
    '/images/prabhas.jpg'
  ];

  // When the user selects a file, store it and generate a preview URL.
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    console.log("Selected file:", file);
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  // When the user clicks "Upload", send the file to the Flask backend.
  const handleSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true); // Show loading indicator
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Axios automatically sets the correct multipart/form-data boundary.
      const response = await axios.post('/api/predict', formData);
      console.log(response.data.message);
      setPrediction(`Prediction: ${response.data.prediction}`);
    } catch (error) {
      console.error('Error uploading file', error);
      if (error.response && error.response.data && error.response.data.error) {
        setPrediction(`Error: ${error.response.data.error}`);
      } else {
        setPrediction('An unknown error occurred.');
      }
    } finally {
      setLoading(false); // Hide loading indicator
    }
  };

  return (
    <div className='container'>
      <header className='header'>
        <h1>Who’s in the Image?</h1>
        <p>Upload any celebrity photo displayed on the screen, and our CNN model will identify who it is!</p>
      </header>
      <div className='image-grid'>
        {images.map((img, index) => (
          <img
            key={index}
            src={img}
            alt={`Option ${index + 1}`}
            className='image'
            style={{ width: '200px', height: '200px', objectFit: 'contain', borderRadius: '10px' }}
          />
        ))}
      </div>
      <div className='upload-section'>
        <input type='file' id='file-upload' onChange={handleFileChange} hidden />
        <label htmlFor='file-upload' className='upload-btn'>📂 Choose a File</label>
        {preview && (
          <img
            src={preview}
            alt='Preview'
            className='mini-preview'
            style={{ width: '50px', height: '50px', objectFit: 'cover', borderRadius: '5px', margin: '10px 0' }}
          />
        )}
        <button className='btn' style={{ marginTop: '10px' }} onClick={handleSubmit}>
          Upload
        </button>
      </div>
      {loading ? (
        <h2 className='prediction'>Loading...</h2>
      ) : (
        <h2 className='prediction'>{prediction}</h2>
      )}
      <footer className='footer'>
        <p>&copy; 2025 Image Classifier | Designed with CNN</p>
      </footer>
    </div>
  );
}

export default App;
