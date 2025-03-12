import streamlit as st
import tempfile
from pathlib import Path
import logging
import asyncio
import json
import nest_asyncio

from resume_processor import ResumeProcessor
from google_form_handler import GoogleFormHandler
from rag_workflow_with_human_feedback import RAGWorkflowWithHumanFeedback
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable nested event loops
nest_asyncio.apply()

def initialize_session_state():
    """Initialize all session state variables"""
    if 'resume_processor' not in st.session_state:
        st.session_state.resume_processor = None  # Initialize as None first
    if 'form_handler' not in st.session_state:
        st.session_state.form_handler = None
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    if 'workflow_handler' not in st.session_state:
        st.session_state.workflow_handler = None
    if 'resume_processed' not in st.session_state:
        st.session_state.resume_processed = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'form_data' not in st.session_state:
        st.session_state.form_data = None
    if 'filled_form' not in st.session_state:
        st.session_state.filled_form = None
    if 'resume_index_path' not in st.session_state:
        st.session_state.resume_index_path = None
    if 'event_loop' not in st.session_state:
        st.session_state.event_loop = None
    if 'openrouter_key' not in st.session_state:
        st.session_state.openrouter_key = None
    if 'llama_cloud_key' not in st.session_state:
        st.session_state.llama_cloud_key = None
    if 'final_form_filled' not in st.session_state:
        st.session_state.final_form_filled = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    # Add feedback-specific state variables
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'current_feedback' not in st.session_state:
        st.session_state.current_feedback = None
    if 'feedback_count' not in st.session_state:
        st.session_state.feedback_count = 0
    if 'last_event_type' not in st.session_state:
        st.session_state.last_event_type = None

# Define available OpenRouter models
OPENROUTER_MODELS = {
    "Mistral 7B Instruct": "mistralai/mistral-7b-instruct:free",
    "DeepSeek R1": "deepseek/deepseek-r1-zero:free",
    "MythoMax L2 13B": "gryphe/mythomax-l2-13b",
    "Llama 2 70B": "meta-llama/llama-2-70b-chat:free",
    "Claude 2.1": "anthropic/claude-2.1",
    "GPT-4": "openai/gpt-4",
    "GPT-3.5 Turbo": "openai/gpt-3.5-turbo"
}

def process_resume(file_input: str) -> bool:
    """Process resume and create index"""
    try:
        with st.spinner("Processing your resume..."):
            result = st.session_state.resume_processor.process_file(file_input)
            print(result)
            if result["success"]:
                st.session_state.resume_index_path = result["index_location"]
                st.success(f"Successfully processed resume! Created {result['num_nodes']} searchable sections.")
                return True
            else:
                error_msg = result["error"]
                if "503 Service Temporarily Unavailable" in error_msg:
                    st.error("""
                    The resume processing service is temporarily unavailable. 
                    Please try again in a few minutes.
                    
                    If the issue persists, you can try:
                    1. Checking your internet connection
                    2. Waiting a few minutes and trying again
                    3. Using a different resume file
                    """)
                else:
                    st.error(f"Failed to process resume: {error_msg}")
                return False
    except Exception as e:
        error_msg = str(e)
        if "503 Service Temporarily Unavailable" in error_msg:
            st.error("""
            The resume processing service is temporarily unavailable. 
            Please try again in a few minutes.
            
            If the issue persists, you can try:
            1. Checking your internet connection
            2. Waiting a few minutes and trying again
            3. Using a different resume file
            """)
        else:
            st.error(f"Error processing resume: {error_msg}")
        return False

def display_progress_bar():
    """Display progress bar based on current step"""
    steps = ["Upload Resume", "Process Form", "Review & Feedback", "Submit"]
    progress = st.session_state.current_step / (len(steps) - 1)
    st.progress(progress)
    st.caption(f"Step {st.session_state.current_step + 1} of {len(steps)}: {steps[st.session_state.current_step]}")

async def run_workflow(form_data):
    """Run the RAG workflow with human feedback"""
    try:
        if not st.session_state.resume_index_path:
            st.error("Resume index not found. Please process your resume again.")
            return None

        logger.info("Starting workflow with resume index: %s", st.session_state.resume_index_path)
        logger.info("Form data: %s", form_data)

        # Create workflow instance if needed
        if st.session_state.workflow is None:
            st.session_state.workflow = RAGWorkflowWithHumanFeedback(timeout=1000, verbose=True)
            logger.info("Created new workflow instance")

        # Create handler if needed or if feedback was just submitted
        if st.session_state.workflow_handler is None or st.session_state.feedback_submitted:
            st.session_state.workflow_handler = st.session_state.workflow.run(
                resume_index_path=st.session_state.resume_index_path,
                form_data=form_data,
                openrouter_key=st.session_state.openrouter_key,
                llama_cloud_key=st.session_state.llama_cloud_key,
                selected_model=st.session_state.selected_model
            )
            logger.info("Created new workflow handler")
            # Reset feedback submitted flag
            st.session_state.feedback_submitted = False

        try:
            async for event in st.session_state.workflow_handler.stream_events():
                logger.info("Received event type: %s", type(event).__name__)
                st.session_state.last_event_type = type(event).__name__
                
                if isinstance(event, InputRequiredEvent):
                    logger.info("Processing InputRequiredEvent")
                    result_data = event.result
                    
                    # Store form data
                    st.session_state.filled_form = result_data
                    
                    # Display the form data
                    st.subheader("📝 Form Responses")
                    
                    # Display each answer
                    for answer in result_data["display"]["answers"]:
                        with st.expander(f"Question: {answer['question']}", expanded=True):
                            st.write("**Entry ID:** ", answer["entry_id"])
                            st.write("**Answer:** ", answer["answer"])
                            st.divider()
                    
                    # Create unique keys for feedback
                    feedback_key = f"feedback_{st.session_state.feedback_count}"
                    submit_key = f"submit_{feedback_key}"
                    
                    # Get feedback
                    feedback = st.text_area(
                        "Review the filled form and provide any feedback:",
                        key=feedback_key,
                        help="If the answers look correct, just write 'OK'. Otherwise, provide specific feedback for improvement."
                    )
                    
                    # Handle feedback submission
                    if st.button("Submit Feedback", key=submit_key):
                        if not feedback:
                            st.warning("Please provide feedback before submitting.")
                            continue
                            
                        logger.info(f"Submitting feedback #{st.session_state.feedback_count}: {feedback}")
                        st.session_state.current_feedback = feedback
                        st.session_state.feedback_submitted = True
                        st.session_state.feedback_count += 1
                        
                        # Send feedback event
                        await st.session_state.workflow_handler.ctx.send_event(
                            HumanResponseEvent(
                                response=feedback
                            )
                        )
                        st.rerun()
                    
                    # Break to prevent multiple feedback forms
                    break
                
            # Only process final result if no feedback is pending
            if not st.session_state.feedback_submitted:
                final_result = await st.session_state.workflow_handler
                logger.info("Workflow completed with final result: %s", final_result)
                
                if final_result and isinstance(final_result, str):
                    try:
                        parsed_result = json.loads(final_result)
                        logger.info("Successfully parsed final result")
                        st.session_state.final_form_filled = parsed_result
                        st.session_state.current_step += 1
                        # Clear workflow state
                        st.session_state.workflow = None
                        st.session_state.workflow_handler = None
                        st.session_state.feedback_count = 0
                        return final_result
                    except json.JSONDecodeError as e:
                        logger.error("Error parsing final result: %s", str(e))
                        st.error("Error processing the final form data. Please try again.")
                        return None
                else:
                    logger.error("Invalid final result format")
                    st.error("Invalid form data received. Please try again.")
                    return None
                    
        except asyncio.CancelledError:
            logger.warning("Workflow was cancelled")
            return None
            
    except Exception as e:
        logger.error("Workflow error: %s", str(e), exc_info=True)
        st.error(f"Error in workflow: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Resume Form Filler",
        page_icon="📝",
        layout="wide"
    )
    
    # Initialize session state first
    initialize_session_state()
    
    # Add API key inputs and guidelines in sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Keys Setup")
        
        # OpenRouter API Key
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Required for AI processing. Get your key at https://openrouter.ai/keys"
        )
        if openrouter_key:
            st.session_state.openrouter_key = openrouter_key
            
        # Llama Cloud API Key
        llama_cloud_key = st.text_input(
            "Llama Cloud API Key",
            type="password",
            help="Required for resume parsing. Get your key at https://cloud.llamaindex.ai/"
        )
        if llama_cloud_key:
            st.session_state.llama_cloud_key = llama_cloud_key
            # Initialize or update resume processor with new API key
            st.session_state.resume_processor = ResumeProcessor(
                storage_dir="resume_indexes",
                llama_cloud_api_key=llama_cloud_key
            )
            
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        selected_model_name = st.selectbox(
            "Choose AI Model",
            options=list(OPENROUTER_MODELS.keys()),
            help="""Select the AI model to use for processing your resume. 
            Free models are marked with ':free'. 
            Paid models may provide better results but require credits."""
        )
        if selected_model_name:
            st.session_state.selected_model = OPENROUTER_MODELS[selected_model_name]
            # Show model info
            st.info(f"""Selected model: {selected_model_name}
            Model ID: {st.session_state.selected_model}
            {'🆓 This is a free model' if ':free' in st.session_state.selected_model else '💰 This is a paid model'}""")
            
        st.markdown("### 📋 How to Get API Keys")
        st.markdown("""
        **OpenRouter API Key:**
        1. Visit [OpenRouter](https://openrouter.ai/)
        2. Sign up or log in
        3. Go to API Keys section
        4. Create a new key
        
        **Llama Cloud API Key:**
        1. Visit [Llama Cloud](https://cloud.llamaindex.ai/)
        2. Create an account
        3. Navigate to API Keys
        4. Generate a new key
        """)
        
        st.markdown("### ⚠️ Important Limitations")
        st.markdown("""
        - Maximum 10 form questions supported
        - PDF files only (max 10MB)
        - Processing time increases with form complexity
        - Ensure stable internet connection
        - API keys are required for all features
        
        **Best Practices:**
        - Use clear, single-page resumes
        - Verify form fields before submission
        - Review AI-generated answers carefully
        - Provide feedback for better results
        """)
        
        st.markdown("### How it works:")
        st.markdown("""
        1. **Upload Resume**: Upload your resume or provide a Google Drive link
        2. **Process Form**: Enter the Google Form URL and review the fields
        3. **Review & Feedback**: Review the auto-filled form and provide feedback
        4. **Submit**: Final review and submission
        
        ### Features:
        - PDF & Google Drive support
        - AI-powered information extraction
        - Interactive feedback system
        - Progress tracking
        - Error handlingʼ
        """)
    
    st.title("📝 Automatic Job Application Form Filler")
    st.write("""
    Upload your resume and provide a Google Form link. 
    This app will automatically extract information from your resume and fill out the form!
    """)
    
    # Check for API keys and model selection before proceeding
    if not st.session_state.openrouter_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to continue.")
        return
    if not st.session_state.llama_cloud_key:
        st.warning("⚠️ Please enter your Llama Cloud API key in the sidebar to continue.")
        return
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select an AI model from the sidebar to continue.")
        return
    
    # Display progress bar
    display_progress_bar()
    
    # Step 1: Resume Upload
    if st.session_state.current_step == 0:
        st.header("Step 1: Upload Resume")
        resume_source = st.radio(
            "Choose resume source:",
            ["Upload PDF", "Google Drive Link"]
        )
        
        if resume_source == "Upload PDF":
            uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
            if uploaded_file:
                # Check file size
                if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                    st.error("File size exceeds 10MB limit. Please upload a smaller file.")
                    return
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    resume_path = tmp_file.name
                if process_resume(resume_path):
                    st.session_state.resume_processed = True
                    st.session_state.current_step += 1
                    st.rerun()
                Path(resume_path).unlink()
                
        else:  # Google Drive Link
            drive_link = st.text_input("Enter Google Drive link to your resume:")
            if drive_link and st.button("Process Resume"):
                if process_resume(drive_link):
                    st.session_state.resume_processed = True
                    st.session_state.current_step += 1
                    st.rerun()
    
    # Step 2: Form Processing
    elif st.session_state.current_step == 1:
        st.header("Step 2: Process Form")
        form_url = st.text_input("Enter Google Form URL:")
        
        if form_url:
            try:
                with st.spinner("Analyzing form fields..."):
                    form_handler = GoogleFormHandler(url=form_url)
                    questions_df = form_handler.get_form_questions_df(only_required=False)
                    
                    # Check number of questions
                    if len(questions_df) >= 20:
                        st.error("⚠️ This form has more than 20 questions. Currently, we only support forms with up to 10 questions for optimal performance.")
                        return
                        
                    st.session_state.form_data = questions_df.to_dict(orient="records")
                    st.session_state.form_url = form_url
                    
                    # Display form fields preview
                    st.subheader("Form Fields Preview")
                    st.dataframe(questions_df)
                    
                    # Show processing time estimate
                    est_time = len(questions_df) * 15  # Rough estimate: 15 seconds per question
                    st.info(f"ℹ️ Estimated processing time: {est_time} seconds")
                    
                    if st.button("Continue to Review"):
                        st.session_state.current_step += 1
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error processing form: {str(e)}")
    
    # Step 3: Review & Feedback
    elif st.session_state.current_step == 2:
        st.header("Step 3: Review & Feedback")
        
        if st.session_state.form_data:
            logger.info("Current form data: %s", st.session_state.form_data)
            logger.info("Current filled form state: %s", st.session_state.filled_form)
            
            # Create new event loop for async operations
            if st.session_state.event_loop is None:
                st.session_state.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(st.session_state.event_loop)
            
            # Run the workflow
            st.session_state.final_form_filled = st.session_state.event_loop.run_until_complete(run_workflow(st.session_state.form_data))
            logger.info("Workflow result: %s", st.session_state.final_form_filled)
    
    # Step 4: Final Submission
    elif st.session_state.current_step == 3:
        st.header("Step 4: Submit Application")
        
        logger.info("Final submission step - Current filled form: %s", st.session_state.filled_form)
        
        if not st.session_state.filled_form:
            st.error("No form data available. Please go back and complete the previous steps.")
            if st.button("Go Back"):
                st.session_state.current_step = 2
                st.rerun()
        else:
            try:
                form_data = st.session_state.filled_form["submission"]
                logger.info("Submission data: %s", form_data)
                
                # Display final review
                st.subheader("📋 Final Review")
                for entry_id, answer in form_data.items():
                    # Find the corresponding question from the original form data
                    question = next(
                        (field["Question"] for field in st.session_state.form_data if field["Entry_ID"] == entry_id),
                        "Unknown Question"
                    )
                    with st.expander(f"Field: {question}", expanded=True):
                        st.write("**Entry ID:** ", entry_id)
                        st.write("**Answer:** ", answer)
                        st.divider()
                
                if st.button("Submit Application", type="primary"):
                    try:
                        # Submit the form using the form handler
                        logger.info("Attempting to submit form to URL: %s", st.session_state.form_url)
                        form_handler = GoogleFormHandler(url=st.session_state.form_url)
                        success = form_handler.submit_form(form_data)
                        
                        if success:
                            st.success("🎉 Application submitted successfully!")
                            st.balloons()
                            logger.info("Form submitted successfully")
                        else:
                            st.error("Failed to submit application. Please try again.")
                            logger.error("Form submission failed without exception")
                    except Exception as e:
                        st.error(f"Error submitting application: {str(e)}")
                        logger.error("Form submission error: %s", str(e), exc_info=True)
            except Exception as e:
                st.error(f"Error preparing submission: {str(e)}")
                st.text("Raw form data:")
                st.json(st.session_state.filled_form)
                logger.error("Error in final submission preparation: %s", str(e))

if __name__ == "__main__":
    main() 