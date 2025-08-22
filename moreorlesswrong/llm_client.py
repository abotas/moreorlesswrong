"""Centralized OpenAI client configuration."""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

# Create a single shared client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))