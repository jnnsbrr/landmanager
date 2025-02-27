"""LLM fertilization component for the model."""

import os
import openai
from dotenv import load_dotenv
from landmanager.components import management

# Load environment variables from .env
load_dotenv()


class Component(management.Component):
    """Model mixing component class for management."""

    def __init__(self, llm, temperature, *args, **kwargs):
        """Initialize the management component."""
        super().__init__(*args, **kwargs)

        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.llm_client = openai.OpenAI(api_key=api_key)
        self.llm_name = llm
        self.temperature = temperature
