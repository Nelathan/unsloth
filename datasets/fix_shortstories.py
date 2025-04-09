from calendar import c
from math import log
import re
import requests
import json
import os
from datasets import load_dataset
from tqdm import tqdm
import logging
import backoff
import sqlite3
import random
from openai import OpenAI
from pathlib import Path
from datasets import Dataset
from google import genai
from google.genai import types
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='spelling_correction.log'
)
logger = logging.getLogger(__name__)

local_api_url = "http://localhost:1234/v1/chat/completions"
temperature=1
threshold=0.25

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

deepseek_client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=os.environ.get("DEEPSEEK_API_KEY")
)

chatgpt_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_PAID"),
)

# models = chatgpt_client.models.list()
# for model in models:
#     logger.info(f"Model: {model}")

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def setup_database():
    conn = sqlite3.connect('synthetic-sugar-quill.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS corrections (
        id INTEGER PRIMARY KEY,
        original_id TEXT,
        original_text TEXT,
        corrected_text TEXT,
        status TEXT,
        api_used TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    return conn


# Improved system prompt
system_editor = """You are an expert literary editor, specializing in enhancing stories with a distinctive and authentic voice, skillfully balancing preservation and improvement.

Your task:
- Enhance the original story's overall quality, while remaining consistent with the authors perspective and vocabulary.
- Correct spelling, grammar, punctuation, and typographical errors.
- Remove metadata, disclaimers, preludes, introductions, dedications, and author’s notes.
- Apply best-practice literary formatting to ensure readability and visual appeal.
- Do not censor explicit content, mature themes, or strong language.
- Preserve the unique style, tone, vision and essence.
- Dont summarize, simplify, or change the plot.
- Optimize: impact, flow, coherence, immersive, prose, imagery, emotional depth, rythm, voice, subtext, tension, character dynamics, mystery, themes, arc
- Minimize: contrived, thin characters, expository, meandering, unearned transformation, overdramatic, unrefined, predictable
- Focus on refinement, not rewrite.

Return only the fully corrected story, with no commentary or extra messages.
"""

system_analyst = """As a literary analyst, read the story and create the authors profile highlighting literary traits.

- Write the profile in first person, as the author, dont repeat yourself, in present tense
- Evaluate skill level
- Identify strengths (voice, themes, devices).
- Pinpoint weaknesses (plot, character development, pacing).
- Infer preferences (genre, tone, style).
- the profile should not reference the story, but stand for itself
- You may consider some of these aspects:
  - Core: impact, flow, fresh, coherent, immersive
  - Finesse: prose, imagery, emotional depth, rhythm, voice, subtext
  - Execution: tension, character dynamics, mystery, themes, arc
  - Flaws: contrived, thin characters, expository, meandering, unearned transformations, overdramatic, unrefined, predictable

Return only the authors profile, with no commentary or extra messages.
"""

# for the next iteration lets use a improved version of the system prompt. The output should look like this:
# <thinking>write authors notes on vision, tone, and style here. Define setting, authors alignment etc</thinking> # this helps the autoregressive model make sense, why it is writing this way, its not random. as an author you always need a vision first, then you can write. this is the vision for the AI
# here goes the fixed short story
# <thinking>reflect on the story, what is good, what could be improved, what is bad</thinking> # this is the reflection part, where the AI can learn from its mistakes and improve
# <rating>integer from 0-9</rating> # the AI learns what a good story is and what a bad story is

# Function to correct text using OpenAI API
def correct_with_openrouter(text):
    model = random.choice([
        # "mistralai/mistral-small-24b-instruct-2501:free",
        # "cognitivecomputations/dolphin3.0-mistral-24b:free", # context limited
        # "google/gemini-2.0-flash-thinking-exp:free", # bad at fixing
        # "google/gemini-2.0-flash-exp:free",
        # "deepseek/deepseek-chat:free",
        "meta-llama/llama-3.3-70b-instruct:free", # slow
        # "google/gemma-2-9b-it:free", # censored
        # Paid models - uncomment if needed
        # "meta-llama/llama-3.3-70b-instruct"
    ])

    try:
        completion = openrouter_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_editor},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=8192,
            temperature=temperature,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
            if (completion.model_dump().get('error')):
                logger.error(f"API error: {completion.model_dump().get('error')}")
                raise ValueError(completion.model_dump().get('error'))
            logger.warning(f"Failure in API response: {completion}")
            raise ValueError("Invalid response structure from API")

        usage = completion.usage

        return completion.choices[0].message.content, model, {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException: {str(e)}")
        raise
    except json.decoder.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def correct_with_deepseek(text):
    model = random.choice([
        "deepseek-chat",
    ])

    completion = deepseek_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_editor},
            {"role": "user", "content": text}
        ],
        stream=False,
        max_completion_tokens=8192,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # if content is None, it means the model failed to correct the text
    if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
        logger.warning(f"Failure in API response: {completion}")
        raise ValueError("Invalid response structure from API")

    usage = completion.usage

    return completion.choices[0].message.content, model, {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None
    }

def correct_with_chatgpt(text):
    model = "gpt-4o-2024-11-20"

    completion = chatgpt_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_editor},
            {"role": "user", "content": text}
        ],
        stream=False,
        max_completion_tokens=8192,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # if content is None, it means the model failed to correct the text
    if not completion or not completion.choices or not completion.choices[0].message or not completion.choices[0].message.content:
        logger.warning(f"Failure in API response: {completion}")
        raise ValueError("Invalid response structure from API")

    usage = completion.usage

    return completion.choices[0].message.content, model, {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None
    }

def correct_with_gemini(text):
    model = random.choice([
        "gemini-2.5-pro-exp-03-25",
        # "gemini-2.0-flash-exp",
        # "gemini-2.0-flash"
    ])
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.90,
        top_k=64,
        # max_output_tokens=8192,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        system_instruction=[
            types.Part.from_text(
                text=system_editor
            ),
        ],
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ],
    )

    response = gemini_client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=text
                    ),
                ],
            ),
        ],
        config=generate_content_config,
    )
    usage = response.usage_metadata

    # Check for prompt feedback (e.g., blocked content)
    if hasattr(response, "prompt_feedback") and response.prompt_feedback:
        logger.warning(f"Blocked because: {response.prompt_feedback}")
        return response.prompt_feedback, model, {
            "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
        }

    # Ensure candidates exist
    if not response.candidates or len(response.candidates) == 0:
        raise ValueError("No candidates returned in the response")

    return response.text, model, {
        "prompt_tokens": usage.prompt_token_count if usage else None,
        "completion_tokens": usage.candidates_token_count if usage else None
    }

def correct_with_groq(text):
    model = random.choice([
        "llama-3.3-70b-specdec",
        # "llama-3.3-70b-versatile",
        # "llama3-70b-8192",
    ])
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_editor},
            {"role": "user", "content": text}
        ],
        model=model,
        max_completion_tokens=8192,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # if content is None, it means the model failed to correct the text
    if not chat_completion.choices or not chat_completion.choices[0].message or not chat_completion.choices[0].message.content:
        raise ValueError(f"Invalid response structure from API: {chat_completion}")

    usage = chat_completion.usage

    return chat_completion.choices[0].message.content, model, {
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None}

def correct_with_local_api(api_url, text):
    model = "phi-4"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_editor},
            {"role": "user", "content": text}
        ],
        # "max_tokens": 8192,
        "temperature": temperature,
        "top_p": 0.9,
        "min_p": 0,
        "repetition_penalty": 0,
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("error"):
            logger.error(f"{result['error']}")
            raise ValueError("API error")
        if not result["choices"] or not result["choices"][0]["message"] or not result["choices"][0]["message"]["content"]:
            logger.warning(f"Cant parse API response: {result}")
        return result["choices"][0]["message"]["content"], model, result["usage"]
    except Exception as e:
        logger.error(f"Local API error: {str(e)}")
        raise

# Function to attempt text correction using multiple backends
def correct_text(text):
    try:
        return random.choice([
            correct_with_deepseek,
            correct_with_openrouter,
            correct_with_groq,
            correct_with_gemini,
            correct_with_chatgpt,
        ])(text)
    except Exception as remote_error:
        logger.error(f"API ERROR: {str(remote_error)}")
        return str(remote_error), "error", None

    # try:
    #     return correct_with_local_api(local_api_url, text)
    # except Exception as local_error:
    #     logger.error(f"Local API failed: {str(local_error)}")

    return None, "none", None

# Main processing function
def process_dataset():
    # Setup
    conn = setup_database()
    cursor = conn.cursor()

    # Load dataset
    dataset = load_dataset("allura-org/sugarquill-10k", split="train")

    # Check for entries already processed if resuming
    processed_ids = set()
    cursor.execute("SELECT original_id FROM corrections WHERE status = 'success' or status = 'bad' or status = 'failure'")
    processed_ids = {row[0] for row in cursor.fetchall()}
    logger.info(f"Resuming: {len(processed_ids)} entries already processed")

    entries = []
    for i, item in enumerate(dataset):
        if str(i) in processed_ids:
            continue

        text = item["text"]
        entries.append((i, len(text), text))

    # entries.sort(key=lambda x: x[1]) # shortest first
    entries.sort(key=lambda x: x[1], reverse=True) # longest first
    logger.info(f"Processing {len(entries)} entries")

    for i, length, text in tqdm(entries, desc="Editing stories"):
      try:
          corrected_text, api_used, usage = correct_text(text)

          # check if usage.completion_tokens is within threshold of prompt_tokens
          original_length = usage.get("prompt_tokens", None) if usage else None
          if original_length:
              original_length -= 150
          corrected_length = usage.get("completion_tokens", None) if usage else None
          success = corrected_text and original_length and corrected_length and abs(corrected_length - original_length) < threshold * original_length
          status = "success" if success else "failure"
          cursor.execute(
              "INSERT OR REPLACE INTO corrections (original_id, original_text, corrected_text, status, api_used, original_length, corrected_length, length_diff) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (str(i), text, corrected_text or "", status, api_used, original_length, corrected_length, corrected_length - original_length if corrected_length and original_length else None)
          )
          logger.info(f"Entry {i}: {status} - diff: {corrected_length - original_length if corrected_length and original_length else None}")
          conn.commit()

      except Exception as e:
          logger.error(f"Error processing entry {i}: {str(e)}")

    conn.close()

def analyze_local():
    conn = setup_database()
    cursor = conn.cursor()

    cursor.execute("SELECT original_id, corrected_text FROM corrections WHERE status = 'success' and author is null")

    for row in tqdm(cursor.fetchall(), desc="Analyzing authors"):
        try:
            text = row[1]

            model = "gemma-3-12b-instruct"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_analyst},
                    {"role": "user", "content": text}
                ],
                "temperature": 1,
                "top_p": 0.9,
                "min_p": 0,
            }

            response = requests.post(local_api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("error"):
                logger.error(f"{result['error']}")
                raise ValueError("API error")
            if not result["choices"] or not result["choices"][0]["message"] or not result["choices"][0]["message"]["content"]:
                logger.warning(f"Cant parse API response: {result}")

            author = result["choices"][0]["message"]["content"]

            cursor.execute(
                "UPDATE corrections SET author = ? WHERE original_id = ?",
                (author, row[0])
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Error processing entry {row[0]}: {str(e)}")

    conn.close()

def analyze_gemini():
    conn = setup_database()
    cursor = conn.cursor()

    cursor.execute("SELECT original_id, corrected_text FROM corrections WHERE status = 'success' and author is null ORDER BY id")

    for row in tqdm(cursor.fetchall(), desc="Analyzing authors using gemini"):
        try:
            text = row[1]

            model = random.choice([
                "gemini-2.5-pro-exp-03-25",
                # "gemini-2.0-flash-exp",
                # "gemini-2.0-flash",
                # "gemma-3-27b-it",
            ])
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.90,
                top_k=64,
                system_instruction=[
                    types.Part.from_text(
                        text=system_analyst
                    ),
                ],
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.OFF,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.OFF,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.OFF,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.OFF,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.OFF,
                    ),
                ],
            )

            response = gemini_client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=text
                            ),
                        ],
                    ),
                ],
                config=generate_content_config,
            )

            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                raise ValueError(f"Blocked because: {response.prompt_feedback}")

            if not response.candidates or len(response.candidates) == 0:
                raise ValueError("No candidates returned in the response")

            author = response.text if response.text else "ERROR"

            author = re.sub(r"^## Author.*?\n\n", "", author, flags=re.MULTILINE)
            author = re.sub(r"\n{3,}", "\n\n", author)
            author = author.rstrip()

            cursor.execute(
                "UPDATE corrections SET author = ? WHERE original_id = ?",
                (author, row[0])
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Error processing entry {row[0]}: {str(e)}")

    conn.close()

def create_final_dataset():
    logger.info("Creating final dataset...")
    corrected_data = []
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute("SELECT original_id, corrected_text, api_used, author FROM corrections WHERE status = 'success' order by original_id")
    for row in cursor.fetchall():
      try:
        id = int(row[0])
        text = row[1]
        model = row[2]
        profile = row[3]

        text = re.sub(r"\*(\s\*){2,}", "***", text)
        text = re.sub(r"\*{3,}", "***", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.rstrip()

        profile = re.sub(r"\n{3,}", "\n\n", profile)
        profile = profile.rstrip()

        corrected_data.append({"id": id, "text": text, "profile": profile, "model": model})
      except Exception as e:
        logger.warning(f"Error adding entry to dataset: {str(e)}")
    conn.close()

    corrected_dataset = Dataset.from_list(corrected_data)
    output_dir = Path("datasets")
    corrected_dataset.to_parquet(output_dir / "synthetic-sugar-quill.parquet")

if __name__ == "__main__":
    process_dataset()
    analyze_gemini()
    create_final_dataset()
