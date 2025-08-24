# 🔍 AI Image Classifier

A powerful web-based image classification application built with **Streamlit** and **TensorFlow's MobileNetV2** model. Upload any image and get instant AI-powered predictions with confidence scores!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Features

- **Real-time Image Classification**: Upload images and get instant predictions
- **Pre-trained MobileNetV2**: Uses Google's efficient neural network model
- **1000+ Classes**: Recognizes objects from ImageNet dataset
- **Top-3 Predictions**: Shows the most likely classifications with confidence scores
- **User-friendly Interface**: Clean, responsive Streamlit web interface
- **Multiple Formats**: Supports JPG, JPEG, and PNG images

## 🚀 Live Demo

[Try the live demo here](#) *(Deploy to get live URL)*

## 📸 Screenshots

*Upload an image and see the magic happen!*

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **MobileNetV2** - Pre-trained image classification model
- **Streamlit** - Web application framework
- **OpenCV** - Image processing
- **NumPy** - Numerical computing
- **Pillow (PIL)** - Image handling

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip or uv package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vhakash/Ai-image-classifier.git
   cd Ai-image-classifier
   ```

2. **Install dependencies**
   
   Using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Using uv
   uv run streamlit run main.py
   
   # Or using streamlit directly
   streamlit run main.py
   ```

4. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 💻 Usage

1. **Launch the app** using the installation steps above
2. **Upload an image** by clicking "Choose an image..." button
3. **Select a file** in JPG, JPEG, or PNG format
4. **Click "Classify image"** to get predictions
5. **View results** with top-3 predictions and confidence scores

### Example Output
```
Predictions:
Egyptian cat: 78.45%
Tabby cat: 15.32%
Tiger cat: 4.21%
```

## 🔧 How It Works

1. **Image Upload**: User uploads an image through Streamlit interface
2. **Preprocessing**: Image is resized to 224x224 pixels and normalized
3. **Model Prediction**: MobileNetV2 processes the image
4. **Post-processing**: Results are decoded and ranked by confidence
5. **Display**: Top-3 predictions shown with percentages

### Technical Details

- **Model**: MobileNetV2 pre-trained on ImageNet
- **Input Size**: 224x224x3 (RGB)
- **Output**: 1000 class probabilities
- **Architecture**: Efficient convolutional neural network

## 📁 Project Structure

```
ai-image-classifier/
├── main.py              # Main Streamlit application
├── pyproject.toml       # Project dependencies (uv)
├── uv.lock             # Lock file for reproducible builds
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── .venv/              # Virtual environment (local)
```

## ⚠️ Important Disclaimers & Limitations

### Model Accuracy
- **Not 100% accurate**: The AI model may misclassify images, especially for:
  - Unusual angles or lighting conditions
  - Objects not well-represented in ImageNet training data
  - Abstract or artistic images
  - Multiple objects in one image
- **Confidence scores** don't guarantee correctness - a 95% prediction can still be wrong
- **Best results** with clear, well-lit, single-object photos

### Technical Limitations
- **Image size**: Large images (>10MB) may cause slow processing or crashes
- **Memory usage**: High-resolution images consume significant RAM
- **Processing time**: First prediction may take 10-30 seconds (model loading)
- **Supported formats**: Only JPG, JPEG, and PNG files

### Use Case Recommendations
- ✅ **Good for**: Educational purposes, demos, general object recognition
- ❌ **Not suitable for**: Medical diagnosis, security applications, critical decisions
- 🔍 **Best practice**: Always verify results manually for important use cases

### Performance Tips
- Use images under 5MB for faster processing
- Crop images to focus on the main subject
- Ensure good lighting and clear visibility
- Try different angles if results seem incorrect

## 🧠 Model Information

**MobileNetV2** is a lightweight, efficient neural network designed for mobile and embedded vision applications:

- **Parameters**: ~3.4 million
- **Top-1 Accuracy**: ~71.3% on ImageNet
- **Speed**: Optimized for real-time inference
- **Classes**: 1,000 ImageNet categories

## 🎨 Customization

### Adding New Models
```python
# In load_model() function
model = tf.keras.applications.ResNet50(weights='imagenet')
# or
model = tf.keras.applications.VGG16(weights='imagenet')
```

### Changing UI Theme
```python
# In main() function
st.set_page_config(
    page_title="Custom Title",
    page_icon="🤖",
    layout="wide"  # or "centered"
)
```

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Heroku
```bash
# Add these files:
# Procfile: web: streamlit run main.py --server.port=$PORT
# runtime.txt: python-3.9.18
```

### Docker
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Vhakash**
- GitHub: [@Vhakash](https://github.com/Vhakash)
- Project: [AI Image Classifier](https://github.com/Vhakash/Ai-image-classifier)

## 🙏 Acknowledgments

- Google for the MobileNetV2 architecture
- TensorFlow team for the pre-trained models
- Streamlit for the amazing web framework
- ImageNet dataset contributors

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/Vhakash/Ai-image-classifier/issues) page
2. Create a new issue with detailed description
3. Star ⭐ the repository if you found it helpful!

---

**Made with ❤️ and Python**