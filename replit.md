# Prescription Scanner

## Overview

This is an offline prescription scanner application that extracts and analyzes prescription information from uploaded images. The system combines OCR (Optical Character Recognition) using Tesseract with AI-powered analysis through Ollama and the Gemma2 model. The application operates entirely offline once properly configured, ensuring privacy and security of medical data. It features a web-based interface for easy image uploads and displays both raw OCR text extraction and intelligent AI analysis of prescription contents.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Interface**: Simple HTML templates with embedded CSS and JavaScript
- **Template Engine**: Flask's Jinja2 templating system with base template inheritance
- **User Experience**: Clean, responsive design with drag-and-drop file upload area
- **Client-side Features**: Copy-to-clipboard functionality and basic form validation

### Backend Architecture
- **Web Framework**: Flask application with minimal configuration
- **Image Processing Pipeline**: Multi-stage approach using OpenCV and PIL
  - Image preprocessing with Gaussian blur and adaptive thresholding
  - Multiple threshold techniques (adaptive and Otsu) for optimal text detection
- **OCR Engine**: Pytesseract wrapper for Tesseract OCR
- **AI Analysis**: Integration with local Ollama API for prescription interpretation
- **File Handling**: Secure file uploads with extension validation and size limits (16MB)

### Data Storage Solutions
- **File Storage**: Local filesystem storage for uploaded images in `uploads/` directory
- **No Database**: Stateless application with no persistent data storage
- **Session Management**: Flask sessions with configurable secret key

### Security and Privacy
- **Offline Operation**: Complete local processing without external API calls
- **File Validation**: Strict file type checking and secure filename handling
- **Size Limits**: Upload size restrictions to prevent abuse
- **Local Processing**: All OCR and AI analysis performed locally

## External Dependencies

### System-level Dependencies
- **Tesseract OCR**: Core OCR engine for text extraction from images
- **ImageMagick**: Image manipulation and enhancement utilities
- **OpenCV**: Computer vision library for image preprocessing
- **Python 3.8+**: Runtime environment

### AI/ML Services
- **Ollama**: Local LLM inference server running on port 11434
- **Gemma2 Model**: Language model for prescription analysis and interpretation
- **Local API**: HTTP communication with Ollama service at localhost:11434

### Python Libraries
- **Flask**: Web framework and routing
- **Pytesseract**: Python wrapper for Tesseract OCR
- **PIL/Pillow**: Image processing and manipulation
- **OpenCV (cv2)**: Advanced image preprocessing
- **NumPy**: Numerical operations for image arrays
- **Requests**: HTTP client for Ollama API communication

### Development Tools
- **Werkzeug**: WSGI utilities for secure file handling
- **Setup Scripts**: Automated Ollama configuration and model pulling utilities