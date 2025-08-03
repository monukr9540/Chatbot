import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English'

def get_prompt_template(language):
    """Get the appropriate prompt template based on selected language"""
    base_template = """
        You are a knowledgeable and friendly Samsung smartphone expert who understands both technical terms and everyday language.
        Interpret and respond to queries using common terms while providing accurate information about the Samsung Galaxy S25 Ultra.

        TERM MAPPING (interpret these terms as equivalent):
        - Price/Cost/Value/Worth
        - Processor/Chip/CPU/Chipset
        - RAM/Memory
        - Storage/Space/Capacity
        - Display/Screen/Panel
        - Camera/Lens/Shooter
        - Battery/Power/Juice
        - Charging/Power delivery/Fast charging
        - OS/Software/Interface/UI
        - Protection/Durability/Toughness
        - Features/Capabilities/Functions
        - Design/Build/Construction
        - Updates/Patches/Security
        - Colors/Variants/Finishes
        
        RESPONSE GUIDELINES:

        1. Device Specifications:
           - Start with "Samsung Galaxy S25 Ultra Specifications:"
           - Present as a clear, organized list
           - Group by categories (Display, Performance, Camera, etc.)
           Example:
           "Samsung Galaxy S25 Ultra Specifications:
           • Display: 6.9-inch WQHD+ Dynamic AMOLED 2X, 3120 x 1440, 1-120Hz adaptive
           • Processor: Snapdragon 8 Elite for Galaxy (3nm)
           • RAM: 12GB LPDDR5X"

        2. Feature Descriptions:
           - Start with "[Feature Name] Details:"
           - Group information under clear headings:
             * What it does
             * How it works
             * Benefits
             * Limitations (if any)

        3. Product Comparisons:
           - Start with "Comparing Samsung Galaxy S25 Ultra vs [Other Device]:"
           - Organize by categories:
            * Display: [Display specs comparison]
            * Performance: [Performance comparison]
            * Camera: [Camera comparison]
            * Battery: [Battery comparison]
            * Software: [Software comparison]
            * Price: [Price comparison]
           - Use bullet points for clear differentiation

        4. When user asks which model should they buy:
        
           Analysis Matrix:
           Only give this column of available products only.
            | Criteria          | S25 Ultra | S25+ | S25 | 
            |-------------------|-----------|------|-----|
            | Display           |           |      |     |           
            | Performance       |           |      |     |           
            | Camera            |           |      |     |           
            | Battery           |           |      |     |           
            | Price             |           |      |     |           

            Recommended: [Model Name]

            Key Advantages:
            1. Benefit Highlights:
            • [Benefit 1]
            • [Benefit 2]

            2. Standout Features:
            • [Feature 1]
            • [Feature 2]

            3. Value Proposition:
            • [Value point 1]
            • [Value point 2]

            Justification:
            [Clear explanation of why this model is recommended]

            Alternative Recommendations:
            - Budget Option: [Model Name]
            - Premium Option: [Model Name]
            - Specialized Use: [Model Name]

        5. Missing Information:
           - Specify exactly which aspects are unavailable
           - Focus on available specifications
        
        6. For Specific Attribute Queries (like camera, display, price, etc.):
           1. IF question asks about an attribute (price, camera, etc.) for a category (S25 series, etc.):
           - List that attribute for ALL products in that category
           - Format as a numbered or bulleted list
           Example for "What are the cameras on the S25 series?":
           "Samsung Galaxy S25 Series Camera Specifications:
           • S25 Ultra: 200MP main, 50MP ultrawide, 10MP telephoto (3x), 50MP periscope (5x)
           • S25+: [Camera specs]
           • S25: [Camera specs]"

           2. IF question asks about an attribute for a specific product:
           - Return ONLY that product's attribute value
           - Format: "[Product Name]: [Attribute Value]"
           Example: "Samsung Galaxy S25 Ultra Battery: 5,000mAh with 45W wired charging support"

        Current Question: {query_str}
        
        Context: {context_str}
        """
    
    # Add language-specific ending
    if language == 'Hindi':
        return base_template + '\nProvide a direct and precise response in normal hindi following the guidelines above: """'
    else:
        return base_template + '\nProvide a direct and precise response in english following the guidelines above: """'

def initialize_chatbot():
    """Initialize the chatbot with the latest OpenAI models"""
    try:
        llm = OpenAI(
            model=os.getenv("gptmodel"),
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        embed_model = OpenAIEmbedding(
            model=os.getenv("embmodel"),
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        load_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = load_client.get_collection("quickstart_gpt4")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        return index
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

def create_query_engine(index, language='English'):
    """Create a query engine with enhanced product comparison and recommendation capabilities"""
    template = get_prompt_template(language)
    qa_prompt = PromptTemplate(template)
    
    return index.as_query_engine(
        text_qa_template=qa_prompt,
        similarity_top_k=7,
        response_mode="compact"
    )

def get_response_text(response):
    """Extract just the response text from the LlamaIndex response object"""
    return str(response.response)

def display_question_boxes():
    """Display clickable question boxes in a grid layout"""
    st.markdown("### Quick Questions")
    
    # Define common questions about Samsung Galaxy S25 Ultra
    questions = [
        {"title": "S25 Ultra Specs", "query": "What are the full specifications of the Samsung Galaxy S25 Ultra?"},
        {"title": "Camera System", "query": "Tell me about the camera system on the S25 Ultra"},
        {"title": "Price & Storage", "query": "What are the prices for different storage options of the S25 Ultra?"},
        {"title": "New Features", "query": "What new features does the S25 Ultra have compared to its predecessor?"},
        {"title": "Color Options", "query": "What colors is the Samsung Galaxy S25 Ultra available in?"},
        {"title": "Case Options", "query": "What official cases are available for the Samsung Galaxy S25 Ultra?"}
    ]
    
    # Create a 2x3 grid layout
    cols = st.columns(3)
    for idx, question in enumerate(questions):
        with cols[idx % 3]:
            if st.button(
                question["title"],
                key=f"q_{idx}",
                use_container_width=True,
                help=question["query"]
            ):
                return question["query"]
    
    return None

def main():
    st.set_page_config(
        page_title="Samsung Galaxy S25 Ultra Information System",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("Samsung Galaxy S25 Ultra Information System 📱")
    
    # Language selector in sidebar
    with st.sidebar:
        st.header("Settings")
        selected_language = st.selectbox(
            "Choose Language",
            ["English", "Hindi"],
            key="language_selector"
        )
        
        # Update session state and reinitialize query engine if language changes
        if selected_language != st.session_state.selected_language:
            st.session_state.selected_language = selected_language
            if 'query_engine' in st.session_state:
                del st.session_state.query_engine
    
    # Initialize system
    try:
        if 'query_engine' not in st.session_state:
            with st.spinner("Initializing system..."):
                index = initialize_chatbot()
                st.session_state.query_engine = create_query_engine(index, st.session_state.selected_language)
            st.success("System initialized successfully!")
        
        # Sidebar with instructions
        with st.sidebar:
            st.header("How to Use")
            st.markdown("""
            You can:
            1. Click on the question boxes above
            2. Ask your own questions in the chat
            3. Get detailed product specifications
            4. Compare different Samsung models
            
            Example questions:
            - "What are the display specs of the S25 Ultra?"
            - "Tell me about the ProVisual Engine"
            - "Which carriers does the S25 Ultra work with?"
            - "What's new in the S25 Ultra compared to S24 Ultra?"
            - "Should I buy the S25 Ultra?"
            """)
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display question boxes at the top
        selected_query = display_question_boxes()
        
        # Add some space between boxes and chat
        st.markdown("---")
        
        # Main chat interface
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").markdown(content)
        
        # Process selected query from question boxes
        if selected_query:
            try:
                with st.spinner("Analyzing your question..."):
                    response = st.session_state.query_engine.query(selected_query)
                    response_text = get_response_text(response)
                
                # Display the response
                st.chat_message("user").write(selected_query)
                st.chat_message("assistant").markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": selected_query},
                    {"role": "assistant", "content": response_text}
                ])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try rephrasing your question.")
        
        # Chat input
        if query := st.chat_input("Ask about the Samsung Galaxy S25 Ultra..."):
            st.chat_message("user").write(query)
            
            try:
                with st.spinner("Analyzing your question..."):
                    response = st.session_state.query_engine.query(query)
                    response_text = get_response_text(response)
                
                # Display the response
                st.chat_message("assistant").markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try rephrasing your question.")
    
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.warning("Please check your configuration and try again.")

if __name__ == "__main__":
    main()