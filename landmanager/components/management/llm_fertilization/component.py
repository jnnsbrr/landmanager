"""LLM fertilization component for the model."""

import openai
from landmanager.components import management


class Component(management.Component):
    """Model mixing component class for management."""

    def __init__(self, llm, temperature, *args, **kwargs):
        """Initialize the management component."""
        super().__init__(*args, **kwargs)

        # move to env vars
        API_KEY = os.getenv("OPENAI_API_KEY")

        self.llm_client = openai.OpenAI(api_key=API_KEY)
        self.llm_name = llm
        self.temperature = temperature
