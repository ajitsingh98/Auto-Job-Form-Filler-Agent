# Auto Job Form Filler Agent 🤖

An intelligent Streamlit application that automatically fills out job application forms using AI-powered resume parsing and form filling capabilities.

## Demo ✌️

https://github.com/user-attachments/assets/6ba74023-5f78-4920-821f-b823cebb316a



## Features ✨

- **Resume Processing**: Upload PDF resumes or provide Google Drive links
- **AI-Powered Form Filling**: Automatically extracts relevant information from resumes
- **Interactive Feedback System**: Review and provide feedback on AI-generated responses
- **Multiple AI Model Support**: Choose from various OpenRouter models
- **Google Forms Integration**: Seamlessly works with Google Forms
- **Progress Tracking**: Visual progress indicator for the application process
- **Error Handling**: Robust error handling and user feedback

## Prerequisites 📋

- Python 3.8+
- OpenRouter API Key
- Llama Cloud API Key
- Google Forms URL

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Auto-Job-Form-Filler-Agent.git
cd Auto-Job-Form-Filler-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Get your OpenRouter API key from [OpenRouter](https://openrouter.ai/)
   - Get your Llama Cloud API key from [Llama Cloud](https://cloud.llamaindex.ai/)

## Usage 💡

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Follow the steps in the application:
   - Upload your resume (PDF or Google Drive link)
   - Enter the Google Form URL
   - Review and provide feedback on AI-generated responses
   - Submit the final application

## Supported AI Models 🤖

- Mistral 7B Instruct (Free)
- DeepSeek R1 (Free)
- MythoMax L2 13B
- Llama 2 70B (Free)
- Claude 2.1
- GPT-4
- GPT-3.5 Turbo

## Limitations ⚠️

- Maximum 10 form questions supported
- PDF files only (max 10MB)
- Processing time increases with form complexity
- Requires stable internet connection
- API keys are required for all features

## Best Practices 📝

- Use clear, single-page resumes
- Verify form fields before submission
- Review AI-generated answers carefully
- Provide feedback for better results

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Limitations

1. **Form Complexity**:
   - Maximum 20 form questions supported
   - Limited to standard Google Forms fields
   - No support for file uploads

2. **File Requirements**:
   - PDF files only
   - Maximum 10MB file size
   - Clear, readable format

3. **Technical Requirements**:
   - Stable internet connection
   - Valid API keys
   - Modern web browser

## Future Improvements

1. **Enhanced AI Models**:
   - Support for more AI models
   - Improved information extraction
   - Better context understanding

2. **Additional Features**:
   - Support for more form types
   - Batch application processing
   - Custom field mapping

3. **User Experience**:
   - Mobile optimization
   - Offline support
   - Custom templates

## Acknowledgments 🙏

- [OpenRouter](https://openrouter.ai/) for AI model access
- [Llama Cloud](https://cloud.llamaindex.ai/) for resume parsing
- [Streamlit](https://streamlit.io/) for the web interface
- [Google Forms](https://www.google.com/forms/about/) for form handling

## Support 💬

If you encounter any issues or have questions, please open an issue in the GitHub repository.
