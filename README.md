# 📝 Auto-Job-Form-Filler-Agent

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Revolutionize your job application process! Upload your resume and let AI handle the form filling.

## 🌟 Features

- 🤖 AI-powered resume information extraction
- 📋 Automatic Google Form filling
- 💡 Interactive feedback system
- 🔄 Real-time progress tracking
- 🛡️ Error handling and validation
- 📱 Responsive web interface
- 🔒 Secure API key management

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- API Keys:
  - [OpenRouter API Key](https://openrouter.ai/keys)
  - [Llama Cloud API Key](https://cloud.llamaindex.ai/)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Auto-Job-Form-Filler-Agent.git
cd Auto-Job-Form-Filler-Agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

## 🎯 How It Works

1. **Upload Resume**: 
   - Support for PDF files (up to 10MB)
   - Google Drive link integration
   - AI-powered resume parsing

2. **Process Form**:
   - Enter Google Form URL
   - Automatic field detection
   - Smart form analysis

3. **Review & Feedback**:
   - AI-generated responses
   - Interactive review system
   - Real-time feedback incorporation

4. **Submit**:
   - Automatic form submission
   - Validation checks
   - Success confirmation

## ⚠️ Limitations

- Maximum 10 form questions supported
- PDF files only (max 10MB)
- Processing time varies with form complexity
- Stable internet connection required
- API keys required for all features

## 💡 Best Practices

- Use clear, single-page resumes
- Verify form fields before submission
- Review AI-generated answers carefully
- Provide feedback for better results

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - LlamaIndex for document processing
  - OpenRouter for LLM integration
  - HuggingFace embeddings
- **Form Handling**: Custom Google Form integration
- **Storage**: Local file system with caching

## 🔒 Security

- API keys are never stored permanently
- Secure session management
- Temporary file handling
- No data retention

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for document processing
- [OpenRouter](https://openrouter.ai/) for AI capabilities
- [Streamlit](https://streamlit.io/) for the web interface

## 📧 Contact

Your Name - [@Ajit](https://x.com/bayesian_walker) - sajit9285@gmail.com

Project Link: [https://github.com/ajitsingh98/Auto-Job-Form-Filler-Agent](https://github.com/ajitsingh98/Auto-Job-Form-Filler-Agent)

---

Made with ❤️ by Ajit
