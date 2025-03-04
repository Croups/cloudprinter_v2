from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import date
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
import os
from dotenv import load_dotenv
import nest_asyncio
import requests
import json
import streamlit as st
import time
import re
from openai import OpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from utils.markdown import to_markdown

# Load environment variables
load_dotenv()
nest_asyncio.apply()

# Configure API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLOUDPRINTER_API_KEY = os.getenv("CLOUDPRINTER_API_KEY")
CLOUDPRINTER_BASE_URL = "https://api.cloudprinter.com/cloudcore/1.0/"

# Available products mapping
PRODUCTS = {
    "business card": "businesscard_ss_int_bc_fc",
    "a4": "card_flat_210x297_mm_double_sided_fc_tnr",
    "flyer": "flyer_ds_a4_fc",
    "folder": "folder_2_x_a4_landscape_half_fold_fc",
    "letterhead": "letterheading_ss_a4_1_0",
    "magazine": "magazine_sas_a4_p_fc",
    "photo print": "photo_print_400x600in_102x152mm",
    "photobook coil": "photobook_cw_a4_p_fc",
    "photobook perfect bound": "photobook_pb_a4_p_fc",
    "poster": "poster_a4_fc",
    "puzzle": "puzzle_480x340_mm_500_pieces",
    "calendar": "calendar_wall_int_a4_p_12_single_fc_tnr"
}

# Define the state model to track order parameters
@dataclass
class PrintOrderState:
    product_code: Optional[str] = None
    product_name: Optional[str] = None
    country: Optional[str] = None
    quantity: Optional[int] = None
    conversation_history: List[Dict[str, str]] = None
    ready_for_quote: bool = False
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
    
    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
    
    def update_product_type(self, product_name_or_code: str) -> bool:
        """Update product type by name or code"""
        # Try as a direct product code first
        if product_name_or_code in PRODUCTS.values():
            self.product_code = product_name_or_code
            for name, code in PRODUCTS.items():
                if code == product_name_or_code:
                    self.product_name = name
                    return True
        
        # Try as a product name
        product_name_lower = product_name_or_code.lower()
        for name, code in PRODUCTS.items():
            if name.lower() in product_name_lower or product_name_lower in name.lower():
                self.product_name = name
                self.product_code = code
                return True
                
        return False
    
    def get_missing_parameters(self):
        missing = []
        if not self.product_code:
            missing.append("product type")
        if not self.country:
            missing.append("country for delivery")
        if not self.quantity:
            missing.append("quantity")
        return missing
    
    def is_complete(self):
        return self.product_code and self.country and self.quantity
        
    def update_ready_for_quote(self):
        self.ready_for_quote = self.is_complete()
        return self.ready_for_quote

# Initialize the OpenAI model
model = OpenAIModel("gpt-4o-mini", api_key=OPENAI_API_KEY)

# Create a unified agent that can handle all aspects of the conversation
unified_agent = Agent(
    model=model,
    deps_type=PrintOrderState,
    result_type=str,
    system_prompt="""
    You are a comprehensive printing assistant that can:
    
    1. Help customers understand printing options and products
    2. Gather necessary information for price quotes
    3. Provide accurate price quotes when all information is available
    
    IMPORTANT INSTRUCTIONS:
    
    - If a customer mentions a product and asks about specifications, sizes, materials, 
      or ANY details, IMMEDIATELY use get_product_info to retrieve product details
      
    - If ALL required information is available (product, country, quantity), you may 
      provide a price quote using get_price_quote
      
    - If information is missing, politely ask ONLY for the missing information
    
    - When the customer provides new information about product, country, or quantity,
      update your memory to remember this information
      
    - Maintain a helpful, conversational tone throughout
    
    - Always check what information has already been provided before asking for it again
    
    - Update the ready_for_quote flag when all information is available
    """
)

@unified_agent.system_prompt
def add_available_products(ctx: RunContext[PrintOrderState]) -> str:
    """Add available products to system prompt"""
    products_list = "\n".join(f"- {name}: {code}" for name, code in PRODUCTS.items())
    return f"Available products with their reference codes:\n{products_list}"

@unified_agent.tool
async def get_product_info(ctx: RunContext[PrintOrderState], product_reference:str):
    """Get information about a specific product
    
    This tool should be used ANY time a customer asks about product details, specifications,
    options, sizes, materials, or any other product information.
    """
    # Update the state with the product code
    ctx.deps.update_product_type(product_reference)
    ctx.deps.update_ready_for_quote()  # Update ready_for_quote flag

    url = f"{CLOUDPRINTER_BASE_URL}products/info/"

    payload = json.dumps({
        "apikey": CLOUDPRINTER_API_KEY,
        "reference": product_reference
    })
    
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

@unified_agent.tool
async def get_price_quote(ctx: RunContext[PrintOrderState], product_code:str, country:str, quantity:int):
    """Get a price quote for a specific product, country and quantity
    
    This tool should ONLY be used when all required information (product, country, quantity) is available.
    """
    # Update the state with the parameters
    ctx.deps.update_product_type(product_code)
    ctx.deps.country = country
    ctx.deps.quantity = quantity
    ctx.deps.update_ready_for_quote()  # Update ready_for_quote flag

    url = f"{CLOUDPRINTER_BASE_URL}orders/quote"

    payload = json.dumps({
        "apikey": CLOUDPRINTER_API_KEY,
        "country": country,
        "items": [
            {
            "reference": "ref_id_1234567",
            "product": product_code,
            "count": quantity,
            "options": []
            }
        ]
    })
    
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

@unified_agent.tool
async def check_conversation_state(ctx: RunContext[PrintOrderState]) -> dict:
    """Check the current conversation state and determine what's missing
    
    Use this tool to understand what information has been provided and what is still needed.
    """
    missing_params = ctx.deps.get_missing_parameters()
    
    # Update ready_for_quote flag
    ctx.deps.ready_for_quote = len(missing_params) == 0
    
    return {
        "product_code": ctx.deps.product_code,
        "product_name": ctx.deps.product_name,
        "country": ctx.deps.country,
        "quantity": ctx.deps.quantity,
        "missing_parameters": missing_params,
        "is_complete": ctx.deps.is_complete(),
        "ready_for_quote": ctx.deps.ready_for_quote
    }

@unified_agent.tool
async def format_missing_parameters_message(ctx: RunContext[PrintOrderState]) -> str:
    """Format a friendly message asking for missing parameters
    
    Use this tool to create a polite message requesting only the missing information.
    """
    missing_params = ctx.deps.get_missing_parameters()
    
    if not missing_params:
        return "I have all the information I need to provide you with a quote!"
    
    if len(missing_params) == 1:
        return f"To provide you with a quote, I still need to know the {missing_params[0]}. Could you please provide this information?"
    else:
        formatted_list = ", ".join(missing_params[:-1]) + " and " + missing_params[-1]
        return f"To provide you with a quote, I still need to know the following: {formatted_list}. Could you please provide this information?"

# Set up the Streamlit interface
st.set_page_config(
    page_title="CloudPrinter Assistant",
    page_icon="🖨️",
    layout="centered"
)

st.title("🖨️ CloudPrinter Assistant")
st.markdown("Ask me about printing products, prices, and specifications!")

# Initialize chat history and state in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your CloudPrinter assistant. Ask me about our products, prices, or specifications."}
    ]

if "print_state" not in st.session_state:
    st.session_state.print_state = PrintOrderState()
    
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input for user message
if prompt := st.chat_input("What would you like to know?"):
    # Extract key information from user input
    # Product detection
    product_found = False
    for product_name, product_code in PRODUCTS.items():
        if product_name.lower() in prompt.lower():
            st.session_state.print_state.update_product_type(product_name)
            product_found = True
            break
    
    # If no direct match, look for partial matches
    if not product_found:
        words = prompt.lower().split()
        for word in words:
            if len(word) > 3:  # Only check significant words
                for product_name in PRODUCTS:
                    if word in product_name.lower():
                        st.session_state.print_state.update_product_type(product_name)
                        product_found = True
                        break
                if product_found:
                    break
    
    # Country detection - simplified list
    common_countries = ["germany", "netherlands", "france", "usa", "uk", "spain", "italy", 
                         "united states", "britain", "deutschland", "brazil", "argentina", 
                         "chile", "mexico", "colombia"]
    for country in common_countries:
        if country.lower() in prompt.lower():
            st.session_state.print_state.country = country
            break
    
    # Quantity detection
    quantity_matches = re.findall(r'\b(\d+)\b', prompt)
    if quantity_matches:
        st.session_state.print_state.quantity = int(quantity_matches[0])
    
    # Update ready_for_quote flag
    st.session_state.print_state.update_ready_for_quote()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.print_state.add_message("user", prompt)
    
    # Display user message and show processing indicator
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.write("Thinking...")
        
        try:
            # Process the message with our unified agent
            response = unified_agent.run_sync(
                prompt,
                deps=st.session_state.print_state
            )
            
            bot_response = response.data
            
            # Display response
            message_placeholder.write(bot_response)
            
            # Add bot response to history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.session_state.print_state.add_message("assistant", bot_response)
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.write(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

# Add sidebar with information and state display
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a simple demo of the CloudPrinter API assistant.
    
    It uses a unified AI agent to answer questions about printing products,
    get price quotes, and provide product specifications.
    """)
    
    # Display current state in the sidebar
    st.subheader("Current Order State")
    if "print_state" in st.session_state:
        st.write({
            "Product": st.session_state.print_state.product_name or "Not specified",
            "Country": st.session_state.print_state.country or "Not specified",
            "Quantity": st.session_state.print_state.quantity or "Not specified",
            "Ready For Quote": st.session_state.print_state.ready_for_quote
        })
    
    # Add a button to reset the conversation
    if st.button("Reset Conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your CloudPrinter assistant. Ask me about our products, prices, or specifications."}
        ]
        st.session_state.print_state = PrintOrderState()
        st.session_state.conversation_id = None
        st.rerun() 