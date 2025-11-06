"""
Utilities Module - Configuration and Token Counting
Handles configuration, token counting, and system setup
"""

import os
import sys
import tiktoken
from dotenv import load_dotenv


# Global token counter
total_tokens = 0
tokenizer = None


def setup_environment():
    """
    Setup environment including Windows console encoding and API keys.
    """
    # Fix Windows console encoding for Unicode characters
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    # Load environment variables
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    return anthropic_api_key


def initialize_tokenizer():
    """
    Initialize the tokenizer for Claude Haiku token counting.

    Returns:
        tiktoken.Encoding: Initialized tokenizer
    """
    global tokenizer

    # Claude uses a different tokenizer, but we can approximate with cl100k_base
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except:
        tokenizer = tiktoken.encoding_for_model("gpt-4")

    return tokenizer


def count_tokens(text):
    """
    Count tokens in text using tiktoken (approximation for Claude Haiku).

    Args:
        text: Text to count tokens for

    Returns:
        int: Number of tokens
    """
    global total_tokens, tokenizer

    if tokenizer is None:
        initialize_tokenizer()

    tokens = len(tokenizer.encode(str(text)))
    total_tokens += tokens
    return tokens


def get_total_tokens():
    """
    Get total tokens counted so far.

    Returns:
        int: Total tokens
    """
    global total_tokens
    return total_tokens


def reset_token_count():
    """
    Reset the global token counter.
    """
    global total_tokens
    total_tokens = 0


def get_user_input_for_test_sentences(max_available):
    """
    Prompt user for number of test sentences.

    Args:
        max_available: Maximum number of test sentences available

    Returns:
        int: Number of test sentences to use
    """
    print("Configuration:")
    print("-" * 80)
    user_input = input(
        "Enter number of test sentences (press Enter for default=100): "
    ).strip()

    if user_input == "":
        num_test_sentences = 100
        print(f"Using default: {num_test_sentences} test sentences")
    else:
        try:
            num_test_sentences = int(user_input)
            if num_test_sentences < 1:
                print("Invalid input. Using default: 100 test sentences")
                num_test_sentences = 100
            else:
                print(f"Using: {num_test_sentences} test sentences")
        except ValueError:
            print("Invalid input. Using default: 100 test sentences")
            num_test_sentences = 100

    print()
    print("=" * 80)
    print()

    return num_test_sentences


def calculate_token_cost(total_tokens, model_name="claude-3-haiku-20240307"):
    """
    Calculate estimated API cost for Claude Haiku.

    Args:
        total_tokens: Total number of tokens
        model_name: Model name for cost calculation

    Returns:
        dict: Cost information
    """
    # Claude Haiku pricing: $0.25 per million input tokens, $1.25 per million output tokens
    input_cost = (total_tokens / 1_000_000) * 0.25

    return {
        'total_tokens': total_tokens,
        'input_cost': input_cost,
        'model': model_name,
        'pricing_note': 'Based on Claude Haiku pricing: $0.25/million input tokens'
    }


# Claude Model Configuration
CLAUDE_MODEL = "claude-3-haiku-20240307"
