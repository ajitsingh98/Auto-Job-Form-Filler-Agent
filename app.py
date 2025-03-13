import streamlit as st
import tempfile
from pathlib import Path
import logging
import asyncio
import json
import nest_asyncio
import time

from resume_processor import ResumeProcessor
from google_form_handler import GoogleFormHandler
from rag_workflow_with_human_feedback import RAGWorkflowWithHumanFeedback
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent, StopEvent

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
    if 'form_url' not in st.session_state:
        st.session_state.form_url = None
    # Add feedback-specific state variables
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'current_feedback' not in st.session_state:
        st.session_state.current_feedback = None
    if 'feedback_count' not in st.session_state:
        st.session_state.feedback_count = 0
    if 'last_event_type' not in st.session_state:
        st.session_state.last_event_type = None
    if 'waiting_for_feedback' not in st.session_state:
        st.session_state.waiting_for_feedback = False
    # Add feedback states container
    if 'feedback_states' not in st.session_state:
        st.session_state.feedback_states = {}

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
            # Also reset the workflow handler when creating new workflow
            st.session_state.workflow_handler = None
            # Reset feedback state
            st.session_state.feedback_count = 0
            st.session_state.current_feedback = None
            st.session_state.last_event_type = None
            st.session_state.waiting_for_feedback = False
            st.session_state.feedback_submitted = False

        # Create or get workflow handler
        if st.session_state.workflow_handler is None:
            logger.info("Creating new workflow handler")
            st.session_state.workflow_handler = st.session_state.workflow.run(
                resume_index_path=st.session_state.resume_index_path,
                form_data=form_data,
                openrouter_key=st.session_state.openrouter_key,
                llama_cloud_key=st.session_state.llama_cloud_key,
                selected_model=st.session_state.selected_model
            )
            logger.info("Created new workflow handler")

        # Check if we're waiting for feedback
        if st.session_state.get('waiting_for_feedback', False):
            logger.info("Waiting for feedback from user")
            
            # Create unique keys for feedback
            feedback_key = f"feedback_{st.session_state.feedback_count}"
            submit_key = f"submit_{feedback_key}"
            # Display the form data
            st.subheader("📝 Form Responses")
            
            # Display each answer
            if st.session_state.filled_form and "display" in st.session_state.filled_form and "answers" in st.session_state.filled_form["display"]:
                for answer in st.session_state.filled_form["display"]["answers"]:
                    with st.expander(f"Question: {answer['question']}", expanded=True):
                        st.write("**Entry ID:** ", answer["entry_id"])
                        st.write("**Answer:** ", answer["answer"])
                        st.divider()
            
            # Get feedback
            feedback = st.text_area(
                "Review the filled form and provide any feedback:",
                key=feedback_key,
                help="If the answers look correct, just write 'OK'. Otherwise, provide specific feedback for improvement."
            )
            
            # Show current feedback value
            if feedback:
                st.info(f"Current feedback text: {feedback}")
            
            # Add a container for feedback submission status
            status_container = st.empty()
            
            # Handle feedback submission with a button
            submit_clicked = st.button(
                "Submit Feedback",
                key=submit_key,
                type="primary",
                use_container_width=True
            )
            if submit_clicked:
                if not feedback:
                    status_container.warning("⚠️ Please provide feedback before submitting.")
                else:
                    try:
                        status_container.info("🔄 Processing feedback...")
                        logger.info(f"Submitting feedback #{st.session_state.feedback_count}: {feedback}")
                        
                        # Store current feedback
                        st.session_state.current_feedback = feedback
                        
                        # Mark feedback as submitted
                        st.session_state.feedback_submitted = True
                        st.session_state.waiting_for_feedback = False
                        
                        # Force a rerun to continue the workflow
                        time.sleep(0.5)  # Brief pause to show the message
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"Error preparing feedback: {str(e)}"
                        logger.error(error_msg)
                        status_container.error(f"❌ {error_msg}")
            
            # If feedback is not submitted yet, we need to wait
            if not st.session_state.get('feedback_submitted', False):
                return None
        
        # If feedback was submitted, send it to the workflow
        if st.session_state.get('feedback_submitted', False):
            try:
                logger.info(f"Sending feedback to workflow: {st.session_state.current_feedback}")
                # Send feedback event
                st.session_state.workflow_handler.ctx.send_event(
                    HumanResponseEvent(
                        response=st.session_state.current_feedback
                    )
                )
                # Reset feedback state
                st.session_state.feedback_submitted = False
                st.session_state.feedback_count += 1
                
                # Continue with the workflow
                logger.info("Feedback sent, continuing workflow")
                
            except Exception as e:
                logger.error(f"Error sending feedback: {str(e)}", exc_info=True)
                st.error(f"Error sending feedback: {str(e)}")
                # Reset feedback state
                st.session_state.feedback_submitted = False
                st.session_state.waiting_for_feedback = True
                return None
        
        # Process events
        final_result = None
        try:
            async for event in st.session_state.workflow_handler.stream_events():
                logger.info("Received event type: %s", type(event).__name__)
                st.session_state.last_event_type = type(event).__name__
                
                if isinstance(event, InputRequiredEvent):
                    logger.info("Processing InputRequiredEvent")
                    result_data = event.result
                    
                    # Store form data
                    st.session_state.filled_form = result_data
                    
                    # Set waiting for feedback flag
                    st.session_state.waiting_for_feedback = True
                    
                    # Force a rerun to show the feedback form
                    st.rerun()
                    
                    # This return is just a placeholder, the rerun will interrupt execution
                    return None
                    
                elif isinstance(event, StopEvent):
                    logger.info("Received StopEvent - workflow complete")
                    if hasattr(event, 'result') and event.result is not None:
                        try:
                            # Handle string or dict result
                            if isinstance(event.result, str):
                                try:
                                    final_result = json.loads(event.result)
                                    logger.info("Successfully parsed JSON result")
                                except json.JSONDecodeError:
                                    logger.warning("Result is not valid JSON, using as raw string")
                                    final_result = {"error": "Failed to parse result as JSON", "raw": event.result}
                            elif isinstance(event.result, dict):
                                final_result = event.result
                                logger.info("Result is already a dictionary")
                            else:
                                logger.error(f"Unexpected result type: {type(event.result)}")
                                final_result = {"error": f"Unexpected result type: {type(event.result)}"}
                                
                            logger.info(f"Final result structure: {type(final_result)}")
                            if isinstance(final_result, dict):
                                logger.info(f"Final result keys: {final_result.keys()}")
                            
                            # Update session state with final result
                            st.session_state.filled_form = final_result
                            st.session_state.final_form_filled = final_result
                            st.session_state.current_step += 1
                            
                            # Clear workflow state
                            st.session_state.workflow = None
                            st.session_state.workflow_handler = None
                            st.session_state.waiting_for_feedback = False
                            st.session_state.feedback_submitted = False
                            
                            # Force a rerun to move to the next step
                            st.rerun()
                            
                            return final_result
                            
                        except Exception as e:
                            logger.error(f"Error processing final result: {str(e)}", exc_info=True)
                            st.error(f"Error processing the final form data: {str(e)}")
                            return None
                    else:
                        logger.warning("StopEvent received with no result")
                        st.warning("No final result received. Please try again.")
                        return None
            
            # If we get here, the event stream ended without a StopEvent
            logger.info("Event stream ended without StopEvent")
            
            # Try to get the final result directly from the handler
            try:
                direct_result = await st.session_state.workflow_handler
                logger.info(f"Got direct result from handler: {direct_result}")
                
                if direct_result:
                    if isinstance(direct_result, str):
                        try:
                            final_result = json.loads(direct_result)
                            logger.info("Successfully parsed direct result JSON")
                        except json.JSONDecodeError:
                            logger.warning("Direct result is not valid JSON, using as raw string")
                            final_result = {"error": "Failed to parse direct result as JSON", "raw": direct_result}
                    elif isinstance(direct_result, dict):
                        final_result = direct_result
                        logger.info("Direct result is already a dictionary")
                    else:
                        logger.warning(f"Unexpected direct result type: {type(direct_result)}")
                        final_result = {"error": f"Unexpected direct result type: {type(direct_result)}"}
                    
                    logger.info(f"Direct result structure: {type(final_result)}")
                    if isinstance(final_result, dict):
                        logger.info(f"Direct result keys: {final_result.keys()}")
                    
                    # Update session state
                    st.session_state.filled_form = final_result
                    st.session_state.final_form_filled = final_result
                    st.session_state.current_step += 1
                    
                    # Clear workflow state
                    st.session_state.workflow = None
                    st.session_state.workflow_handler = None
                    st.session_state.waiting_for_feedback = False
                    st.session_state.feedback_submitted = False
                    
                    # Force a rerun to move to the next step
                    st.rerun()
                    
                    return final_result
            except Exception as e:
                logger.error(f"Error getting direct result: {str(e)}", exc_info=True)
            
            # If we still don't have a result, check if we have filled form data
            if st.session_state.filled_form:
                logger.info("Using existing filled form data")
                return st.session_state.filled_form
            
            # If all else fails
            logger.warning("No result available")
            st.warning("No result available. Please try again.")
            return None
                
        except asyncio.CancelledError:
            logger.warning("Workflow was cancelled")
            st.warning("The workflow was cancelled. Please try again.")
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
            result = st.session_state.event_loop.run_until_complete(run_workflow(st.session_state.form_data))
            logger.info("Workflow result: %s", result)
            
            # If we got a final result, store it and move to the next step
            if result and isinstance(result, dict) and "submission" in result:
                st.session_state.filled_form = result
                st.session_state.final_form_filled = result
                
                # Move to next step if we have a final result
                if st.session_state.current_step < 3:
                    st.session_state.current_step = 3
                    st.rerun()
    
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
                form_data = st.session_state.filled_form
                logger.info("Submission data: %s", form_data)
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