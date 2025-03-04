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
from utils.markdown import to_markdown
import streamlit as st
import time
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from utils.markdown import to_markdown
from dotenv import load_dotenv
import nest_asyncio
import os
import json
from dotenv import load_dotenv
from utils.markdown import to_markdown
import requests
import json


# Load environment variables
load_dotenv()
nest_asyncio.apply()

model = "gpt-4o"

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

# Apply nest_asyncio to allow async code in environments like notebooks.
nest_asyncio.apply()
load_dotenv()

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

model = OpenAIModel("gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

api_call_agent = Agent(
    model=model,
    deps_type=PrintOrderState,
    result_type=str,
    system_prompt="""
    You are a helpful api_call agent that can call external API of cloudprinter.com.
    
    IMPORTANT: You must maintain context between user messages. If the user provides
    information about product type, quantity, or country in separate messages, collect
    and remember this information.
    
    CONTEXT AWARENESS:
    1. If the user mentions a product (business card, flyer, etc.), update your context
    2. If the user mentions a country name, update your context
    3. If the user mentions a number that could be a quantity, update your context
    4. Always check what information you already have before asking for it again
    
    If user asks about a product, first call get_product_info tool to get the product information.
    (Never answer product questions without tool calls)
    
    If you have all three pieces of information (product_code, country, quantity), then
    automatically call get_price_quote without explicitly asking the user for more information.
    
    If any parameter is missing, politely ask the user to provide ONLY the missing parameters.
    Do not ask for information that has already been provided.
    
    Return a final answer with the fetched tool response in a user-friendly format.
    Be kind and helpful.
    """
)

# Create a conversation state agent to handle the dialogue flow
conversation_agent = Agent(
    model=model,
    deps_type=PrintOrderState,
    result_type=str,
    system_prompt="""
    You are an assistant that helps gather necessary information for printing orders.
    
    Your task is to analyze the conversation context and determine what information is missing.
    You need to collect three key pieces of information:
    1. Product type (e.g., business card, flyer, poster)
    2. Delivery country
    3. Quantity
    
    IMPORTANT RULES:
    - You must maintain context between user messages
    - If all required information is available, set ready_for_quote to True
    - If any information is missing, set ready_for_quote to False
    - Always be friendly and helpful in your responses
    - Ask only for missing information, don't ask for information already provided
    """
)

@conversation_agent.system_prompt
def get_available_products(ctx: RunContext[PrintOrderState]) -> str:
    """Add available products to system prompt"""
    products_list = "\n".join(f"- {name}" for name in PRODUCTS.keys())
    return f"Available products that we can print:\n{products_list}"

@conversation_agent.tool
async def check_conversation_state(ctx: RunContext[PrintOrderState]) -> dict:
    """Check the current conversation state and determine what's missing"""
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

@conversation_agent.tool
async def format_missing_parameters_message(ctx: RunContext[PrintOrderState]) -> str:
    """Format a message asking for missing parameters"""
    missing_params = ctx.deps.get_missing_parameters()
    
    if not missing_params:
        return "I have all the information I need to provide you with a quote!"
    
    if len(missing_params) == 1:
        return f"To provide you with a quote, I still need to know the {missing_params[0]}. Could you please provide this information?"
    else:
        formatted_list = ", ".join(missing_params[:-1]) + " and " + missing_params[-1]
        return f"To provide you with a quote, I still need to know the following: {formatted_list}. Could you please provide this information?"

@api_call_agent.system_prompt
def add_products(ctx: RunContext[PrintOrderState]) -> str:
    """Add available products to system prompt"""
    products_list = "\n".join(f"- {name}: {code}" for name, code in PRODUCTS.items())
    return f"Available products with their reference codes:\n{products_list}"

@api_call_agent.tool
async def get_price_quote(ctx: RunContext[PrintOrderState], product_code:str, country:str, quantity:int):
    """Get a price quote for a specific product, country and quantity"""
    # Update the state with the parameters using our helper method
    ctx.deps.update_product_type(product_code)
    ctx.deps.country = country
    ctx.deps.quantity = quantity

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
    
@api_call_agent.tool
async def get_product_info(ctx: RunContext[PrintOrderState], product_reference:str):
    """Get information about a specific product"""
    # Update the state with the product code
    ctx.deps.update_product_type(product_reference)

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

# Set page title and layout
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
    import re
    quantity_matches = re.findall(r'\b(\d+)\b', prompt)
    if quantity_matches:
        st.session_state.print_state.quantity = int(quantity_matches[0])
    
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
            # Process conversation state
            conversation_response = conversation_agent.run_sync(
                prompt,
                deps=st.session_state.print_state
            )
            
            # Either get quote or ask for missing information
            if st.session_state.print_state.ready_for_quote:
                # Create prompt for price quote
                price_prompt = f"Please provide a price quote for {st.session_state.print_state.quantity} {st.session_state.print_state.product_name}(s) to be delivered to {st.session_state.print_state.country}."
                response = api_call_agent.run_sync(
                    price_prompt, 
                    deps=st.session_state.print_state
                )
                bot_response = response.data
            else:
                # Use conversation agent response to ask for missing info
                bot_response = conversation_response.data
            
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
    
    It uses AI agents to answer questions about printing products,
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


