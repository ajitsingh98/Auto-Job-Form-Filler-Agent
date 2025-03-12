from llama_parse import LlamaParse
from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
    Context
)
import os
import json
from llama_index.llms.openrouter import OpenRouter
import nest_asyncio
from resume_processor import ResumeProcessor
nest_asyncio.apply()
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    PromptHelper
)
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google_form_handler import GoogleFormHandler
from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent

class QueryEvent(Event):
    query: str

class ParseFormEvent(Event):
    form_data: list

class ResponseEvent(Event):
    response: str

class FeedbackEvent(Event):
    feedback: str

class RAGWorkflowWithHumanFeedback(Workflow):
    
    llm: OpenRouter
    query_engine: VectorStoreIndex

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:

        if not ev.resume_index_path:
            raise ValueError("Resume indexing is required!!")
        
        if not ev.form_data:
            raise ValueError("Form data is required!!")
            
        if not ev.openrouter_key:
            raise ValueError("OpenRouter API key is required!!")
            
        if not ev.llama_cloud_key:
            raise ValueError("Llama Cloud API key is required!!")
            
        if not ev.selected_model:
            raise ValueError("LLM model selection is required!!")
        
        # Configure context window and other settings
        context_window = 4096  # Reduced from 8192 to be safer
        num_output = 2048
        
        # Create prompt helper with appropriate settings
        prompt_helper = PromptHelper(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None
        )

        # Initialize LLM with appropriate settings
        self.llm = OpenRouter(
            api_key=ev.openrouter_key,
            max_tokens=num_output,
            context_window=context_window,
            model=ev.selected_model,
            temperature=0.3,
            top_p=0.9,
        )

        # Test LLM connection
        try:
            test_response = self.llm.complete("Test connection.")
            if not test_response or not test_response.text:
                raise ValueError("LLM returned empty response")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise ValueError("Failed to initialize LLM. Please check your API key and connection.")

        # Configure service context with prompt helper
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2",
            cache_folder="embeddings_cache"  # Cache embeddings for better performance
        )
        Settings.llm = self.llm
        Settings.prompt_helper = prompt_helper
        service_context = Settings

        if os.path.exists(ev.resume_index_path):
            storage_context = StorageContext.from_defaults(persist_dir=ev.resume_index_path)
            index = load_index_from_storage(
                storage_context,
                service_context=service_context
            )
        else:
            raise ValueError("Index not found!!")

        # Configure query engine with better settings
        self.query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=5,
            response_mode="tree_summarize",  # Better for structured responses
            structured_answer_filtering=True,  # Filter out irrelevant information
            response_kwargs={
                "verbose": True,
                "similarity_threshold": 0.7  # Only use highly relevant context
            }
        )

        return ParseFormEvent(form_data=ev.form_data)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent | FeedbackEvent) -> QueryEvent:
        # Get the form data from context if it's a FeedbackEvent
        if isinstance(ev, FeedbackEvent):
            fields = await ctx.get("form_data")
            if not fields:
                raise ValueError("No form data found in context")
        else:
            fields = ev.form_data
            # Store form data in context for future use
            await ctx.set("form_data", fields)
        
        for field in fields:
            question = field["Question"]
            options = field["Options"]
            required = field["Required"]
            entry_id = field["Entry_ID"]
            selection_type = field.get("Selection_Type", "Text")
            
            # Construct a more focused query based on field type
            if selection_type in ["Single Choice", "Dropdown"]:
                query = f"""Based on the candidate's resume, which of these options best answers the following question?
                Question ID: {entry_id}
                Question: {question}
                Options: {options}
                Please select the most appropriate option based on the candidate's experience and qualifications."""
            elif selection_type == "Multiple Choice":
                query = f"""Based on the candidate's resume, which of these options apply to the following question?
                Question ID: {entry_id}
                Question: {question}
                Options: {options}
                Please select all relevant options based on the candidate's experience and qualifications."""
            else:
                query = f"""Based on the candidate's resume, please provide a factual answer to the following question:
                Question ID: {entry_id}
                Question: {question}
                Please be specific and concise in your response."""
            
            if isinstance(ev, FeedbackEvent):
                query += f"""\nWe previously got feedback about how we answered the question. 
                It might be not related to this particular field, but here is feedback.
                <feedback>
                {ev.feedback}
                </feedback>
                """

            ctx.send_event(QueryEvent(
                query=query,
                query_type="Resume Analysis",
                field=question,
                entry_id=entry_id,
                required=required
            ))
        
        await ctx.set("total_fields", len(fields))
        return
        
    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:

        query = f"""Please analyze the following question about the candidate's resume and provide a detailed, factual response.
        Focus on specific details from the resume and maintain a professional tone.
        
        {ev.query}
        
        Guidelines:
        1. Use specific details from the resume when available
        2. If the information is not found in the resume, clearly state that
        3. For multiple choice questions, explain your selection based on the resume content
        """
        
        try:
            # First try using the query engine
            try:
                response = self.query_engine.query(query)
                if response and hasattr(response, 'response') and response.response:
                    response_text = response.response
                else:
                    raise ValueError("Empty response from query engine")
            except Exception as query_error:
                print(f"Query engine error: {str(query_error)}")
                # Fallback to direct LLM if query engine fails
                llm_response = self.llm.complete(query)
                if not llm_response or not llm_response.text:
                    raise ValueError("Empty response from LLM fallback")
                response_text = llm_response.text
                
            print(f"Response: {response_text}")
            
            return ResponseEvent(
                response=response_text,
                field=ev.field,
                entry_id=ev.entry_id,
                required=ev.required
            )
        except Exception as e:
            error_msg = str(e)
            print(f"Error in ask_question: {error_msg}")
            
            # Return a more informative fallback response
            fallback_msg = (
                "Unable to process this question at the moment. "
                "If this is a required field, please try again or provide the information manually."
            )
            
            if ev.required:
                fallback_msg += " (This is a required field)"
            
            return ResponseEvent(
                response=fallback_msg,
                field=ev.field,
                entry_id=ev.entry_id,
                required=ev.required
            )

    @step
    async def fill_in_application(self, ctx:Context, ev:ResponseEvent) -> InputRequiredEvent:
        total_fields = await ctx.get("total_fields")
        responses = ctx.collect_events(ev, [ResponseEvent]*total_fields)
        form_data = await ctx.get("form_data")

        if responses is None:
            return None 

        # Create a structured format for responses
        responsesList = "\n".join(
            f"Entry ID: {r.entry_id}\n" + 
            f"Question: {r.field}\n" + 
            f"Response: {r.response}\n" + 
            f"---" 
            for r in responses
        )        
        
        result = self.llm.complete(f"""
            You are an expert at analyzing resumes and filling out application forms. Your task is to:
            1. Review the questions and responses about a candidate's resume
            2. For each question, provide a clear, concise, and factual answer
            Guidelines:
            - For multiple choice questions, select the most relevant option
            - If a question cannot be answered from the resume, indicate "Not found in resume"
            <responses>
            {responsesList}
            </responses>

            Please provide your response in the following JSON format:
            {{
                "answers": [
                    {{
                        "entry_id": "Entry ID",
                        "question": "Question text",
                        "answer": "Your answer here"
                    }},
                    ...
                ]
            }}

            Important: Ensure the response is valid JSON with double quotes around property names and string values.
        """)
        
        try:
            # Clean up the result text
            result_text = result.text.strip()
            
            # Remove any LaTeX formatting if present
            if '\\boxed{' in result_text:
                start_idx = result_text.find('\\boxed{') + 7
                end_idx = result_text.rfind('}')
                if start_idx > 6 and end_idx > start_idx:
                    result_text = result_text[start_idx:end_idx]
            
            # Try to parse the JSON
            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                # If parsing fails, create a properly formatted JSON
                result_json = {
                    "answers": [
                        {
                            "entry_id": r.entry_id,
                            "question": r.field,
                            "answer": r.response
                        } for r in responses
                    ]
                }
            
            # Create submission data with entry IDs
            submission_data = {}
            for answer in result_json["answers"]:
                entry_id = answer["entry_id"]
                submission_data[entry_id] = answer["answer"]
            
            # Store both display and submission versions
            final_data = {
                "display": result_json,
                "submission": submission_data
            }
            
            # Store in context without JSON encoding
            await ctx.set("filled_form", final_data)
            
            # Return the display version for review without additional JSON encoding
            return InputRequiredEvent(
                prompt="""Please review the filled form and provide any feedback.
                Type your feedback below:""",
                prefix="Your feedback: ",
                request_type="Feedback",
                result=final_data
            )
            
        except Exception as e:
            print(f"Error processing form data: {str(e)}")
            # Fallback to basic structure
            fallback_data = {
                "answers": [
                    {
                        "entry_id": r.entry_id,
                        "question": r.field,
                        "answer": r.response
                    } for r in responses
                ]
            }
            
            # Create submission data for fallback
            submission_data = {}
            for answer in fallback_data["answers"]:
                entry_id = answer["entry_id"]
                submission_data[entry_id] = answer["answer"]
            
            final_data = {
                "display": fallback_data,
                "submission": submission_data
            }
            
            # Store in context without JSON encoding
            await ctx.set("filled_form", final_data)
            
            # Return the display version for review without additional JSON encoding
            return InputRequiredEvent(
                prompt="""Please review the filled form and provide any feedback.
                Type your feedback below:""",
                prefix="Your feedback: ",
                request_type="Feedback",
                result=final_data
            )
    
    @step
    async def get_feedback(self, ctx: Context, ev: HumanResponseEvent) -> FeedbackEvent | StopEvent:
        result = self.llm.complete(f"""
            You have received feedback from a human on the answers you provided. 
            Analyze the feedback and determine if any changes are needed.
            
            <feedback>
            {ev.response}
            </feedback>
            
            Guidelines for response:
            1. If the feedback indicates everything is correct or no changes are needed, respond with "OKAY"
            2. If the feedback suggests any changes or improvements, respond with "FEEDBACK"
            3. Be conservative - if there's any doubt, respond with "FEEDBACK"
            
            Respond with exactly one word: either "OKAY" or "FEEDBACK"
        """)
        
        # Extract just the text content, removing any LaTeX formatting
        verdict = result.text.strip()
        # Remove any LaTeX formatting
        verdict = verdict.replace('\\boxed{', '').replace('}', '')
        # Remove any whitespace and convert to uppercase for consistent comparison
        verdict = verdict.strip().upper()
        
        print(f"LLM says the verdict was {verdict}")
        
        # Get the current filled form data
        filled_form = await ctx.get("filled_form")
        if filled_form:
            try:
                if isinstance(filled_form, str):
                    form_data = json.loads(filled_form)
                else:
                    form_data = filled_form
                # Return the submission data for the final result
                if "OKAY" in verdict:
                    # Ensure we're returning a properly formatted JSON string
                    submission_data = json.dumps(form_data["submission"], ensure_ascii=False, indent=2)
                    return StopEvent(result=submission_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing filled form data: {str(e)}")
                # Return a structured error response
                error_response = {
                    "error": str(e),
                    "message": "Failed to process form data"
                }
                return StopEvent(result=json.dumps(error_response, ensure_ascii=False, indent=2))
        
        return FeedbackEvent(feedback=ev.response)
            
if __name__ == "__main__":
    async def main():
        
        url="https://docs.google.com/forms/d/e/1FAIpQLSchbdsD0MoCCqE8quU3pqQ3zO2qfZxPH_SBjgllfzNhqa-FUQ/viewform"
        form_handler = GoogleFormHandler(url=url)
        # Get form questions as DataFrame
        questions_df = form_handler.get_form_questions_df(only_required=False).head(3)
        form_data = questions_df.to_dict(orient="records")
        processor = ResumeProcessor(storage_dir="resume_indexes")
        # Example with local file
        result = processor.process_file("/Users/ajitkumarsingh/AutoFormAgent/asset/resume.pdf")
        workflow = RAGWorkflowWithHumanFeedback(timeout=1000, verbose=True)
        handler =  workflow.run(
            resume_index_path="resume_indexes",
            form_data=form_data,
            openrouter_key=get_openrouter_api_key(),
            llama_cloud_key=get_llama_cloud_api_key(),
            selected_model="gryphe/mythomax-l2-13b"
        )

        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                print("We'have filled in your form, Here are the results:\n")
                print("Filled form:")
                result_json = event.result

                print(result_json["display"])
                response = input(event.prefix)
                handler.ctx.send_event(
                    HumanResponseEvent(
                        response=response
                    )
                )
        response = await handler
        print("Response:")
        print(response)
    import asyncio
    asyncio.run(main())