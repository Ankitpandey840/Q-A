# -*- coding: utf-8 -*-
"""
RAG-based Question Answering System for Smartphone Data using RoBERTa.
Loads data from a CSV, processes it, and answers questions about specific phones.
Includes flexible model name matching in retrieval.
"""

# ---------------------------------------------------------------------------
# Section 1: Installs and Imports
# ---------------------------------------------------------------------------
import sys
import subprocess
import pkg_resources  # Use pkg_resources for checking installations

# Check and install necessary libraries quietly
print("--- Section 1: Checking/Installing Libraries ---")
required = {'pandas', 'torch', 'transformers', 'sentencepiece', 'openpyxl'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Installing missing packages: {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(missing))
        print("Libraries installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install packages: {e}")
        # Depending on the environment, you might want to stop execution
        # raise RuntimeError("Required packages could not be installed.") from e
else:
    print("All required libraries seem to be installed.")

import pandas as pd
# Set pandas option to handle future downcasting changes if needed (suppresses warning)
pd.set_option('future.no_silent_downcasting', True)  # Optional: Uncomment if you see the warning and want the future behavior
import torch
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
import warnings
import logging
import re  # For retrieval matching
import os  # To check if file exists

print("\n--- Section 1: Imports Complete ---")

# ---------------------------------------------------------------------------
# Section 2: Configuration and Setup
# ---------------------------------------------------------------------------
print("\n--- Section 2: Configuration ---")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress specific warnings from transformers if needed
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# --- !!! IMPORTANT: SET YOUR CSV FILE PATH HERE !!! ---
CSV_FILE_PATH = '/content/sm.csv'
# -----------------------------------------------------

# --- QA Model Configuration ---
QA_MODEL_NAME = "deepset/roberta-base-squad2"
# --------------------------------------------

print(f"Configuration:")
print(f"  - CSV File Path: {CSV_FILE_PATH}")
print(f"  - QA Model Name: {QA_MODEL_NAME}")

# Check if the configured CSV file exists BEFORE proceeding
print(f"\nChecking for CSV file at '{CSV_FILE_PATH}'...")
if not os.path.exists(CSV_FILE_PATH):
    print(f"\n---> FATAL ERROR: File not found at '{CSV_FILE_PATH}'!")
    print("---> Please upload the CSV file to the '/content/' directory and restart the runtime or re-run the cell.")
    # Stop execution if file doesn't exist
    raise FileNotFoundError(f"Required CSV file not found at {CSV_FILE_PATH}")
else:
    print(f"---> File found. Proceeding...")

print("\n--- Section 2: Configuration Complete ---")

# ---------------------------------------------------------------------------
# Section 3: Function Definitions
# ---------------------------------------------------------------------------
print("\n--- Section 3: Defining Functions ---")

def load_data(filepath):
    """Loads and cleans smartphone data from the specified CSV file.
       Also creates a simplified 'base_model_lower' column for matching."""
    print(f"\nAttempting to load data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data from CSV '{filepath}'. Found {len(df)} records.")
        print(f"Successfully loaded raw data from CSV. Found {len(df)} records.")

        print("Starting data cleaning...")

        # --- Numeric Columns ---
        numeric_cols = ['price', 'rating', 'num_cores', 'processor_speed', 'battery_capacity',
                        'fast_charging', 'ram_capacity', 'internal_memory', 'screen_size',
                        'refresh_rate', 'num_rear_cameras', 'num_front_cameras',
                        'primary_camera_rear', 'primary_camera_front', 'extended_upto',
                        'resolution_width', 'resolution_height']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)  # Coerce errors, fill NaN with 0, convert to int

        # --- Boolean Columns ---
        bool_cols = ['has_5g', 'has_nfc', 'has_ir_blaster', 'fast_charging_available', 'extended_memory_available']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().replace({
                    'true': True, '1': True, '1.0': True, 'yes': True, 't': True,
                    'false': False, '0': False, '0.0': False, 'no': False, 'f': False,
                    'nan': False, '': False, 'unknown': False, 'none': False
                }).fillna(False)  # Fill any remaining NaNs
                df[col] = df[col].apply(lambda x: x if isinstance(x, bool) else False)

        # --- String Columns ---
        str_cols = ['processor_brand', 'os', 'brand_name', 'model']
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()  # Fill NaN, ensure string, remove leading/trailing spaces

        # --- Create Base Model Name for Improved Matching ---
        if 'model' in df.columns:
            print("  Creating 'base_model_lower' for improved matching...")
            df['base_model_temp'] = df['model'].str.replace(r'\s*\(.*?\)\s*', '', regex=True).str.strip()
            df['base_model_temp'] = df['base_model_temp'].str.replace(r'\s+5g$', '', regex=True, case=False).str.strip()
            df['base_model_temp'] = df['base_model_temp'].str.replace(r'\s+4g$', '', regex=True, case=False).str.strip()
            df['base_model_lower'] = df['base_model_temp'].str.lower()
            df = df.drop(columns=['base_model_temp'])
        else:
            print("Warning: 'model' column not found, cannot create base model name.")
            df['base_model_lower'] = pd.NA  # Assign Not Available if 'model' column is missing

        print("Data cleaning finished.")
        return df
    except FileNotFoundError:
        logging.error(f"Error during load_data: File not found at '{filepath}'.")
        print(f"ERROR: File not found at '{filepath}' within load_data function.")
        return None
    except Exception as e:
        logging.exception(f"An error occurred while loading or processing the CSV data in load_data:")  # Log traceback
        print(f"ERROR: An error occurred loading/processing the CSV: {e}")
        return None

def create_context_snippets(df):
    """Converts each row of the DataFrame into an individual text description."""
    print("\nCreating context snippets for each phone...")
    if df is None:
        print("ERROR: DataFrame is None, cannot create snippets.")
        return []

    context_snippets = []
    required_cols = ['model', 'brand_name', 'price', 'rating', 'has_5g', 'has_nfc', 'has_ir_blaster',
                     'processor_brand', 'num_cores', 'processor_speed', 'battery_capacity',
                     'fast_charging_available', 'fast_charging', 'ram_capacity', 'internal_memory',
                     'screen_size', 'refresh_rate', 'resolution_width', 'resolution_height',
                     'num_rear_cameras', 'primary_camera_rear', 'num_front_cameras', 'primary_camera_front',
                     'os', 'extended_memory_available', 'extended_upto']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nWarning: The following expected columns are missing from the DataFrame: {missing_cols}")
        print("Context generation might be incomplete or fail for some entries.\n")

    for index, row in df.iterrows():
        model_name = row.get('model', 'N/A')
        brand_name = row.get('brand_name', 'N/A')

        price = int(row.get('price', 0))
        rating = int(row.get('rating', 0))
        num_cores = int(row.get('num_cores', 0))
        proc_speed = row.get('processor_speed', 'N/A')
        battery_cap = int(row.get('battery_capacity', 0))
        ram_cap = int(row.get('ram_capacity', 0))
        int_mem = int(row.get('internal_memory', 0))
        screen_size = row.get('screen_size', 'N/A')
        refresh_rate = int(row.get('refresh_rate', 0))
        res_w = int(row.get('resolution_width', 0))
        res_h = int(row.get('resolution_height', 0))
        num_rear_cam = int(row.get('num_rear_cameras', 0))
        rear_cam_mp = int(row.get('primary_camera_rear', 0))
        num_front_cam = int(row.get('num_front_cameras', 0))
        front_cam_mp = int(row.get('primary_camera_front', 0))
        ext_upto_val = int(row.get('extended_upto', 0))

        has_5g = bool(row.get('has_5g', False))
        has_nfc = bool(row.get('has_nfc', False))
        has_ir = bool(row.get('has_ir_blaster', False))
        fc_avail = bool(row.get('fast_charging_available', False))
        ext_mem_avail = bool(row.get('extended_memory_available', False))

        proc_brand = row.get('processor_brand', 'Unknown')
        os_name = row.get('os', 'Unknown')

        fc_wattage_num = int(row.get('fast_charging', 0))
        fast_charging_info = f"{fc_wattage_num}W" if fc_avail and fc_wattage_num > 0 else ("available" if fc_avail else "not available")
        extended_memory_info = f"Yes, up to {ext_upto_val} GB" if ext_mem_avail and ext_upto_val > 0 else ("Yes" if ext_mem_avail else "No")

        desc = (
            f"Phone: {brand_name} {model_name}. "
            f"Price: {price}. Rating: {rating}. OS: {os_name}. "
            f"Display: {screen_size} inches, {refresh_rate} Hz refresh rate, {res_w}x{res_h} resolution. "
            f"Performance: {proc_brand} processor, {num_cores} cores, {proc_speed} GHz speed, {ram_cap} GB RAM, {int_mem} GB internal memory. "
            f"Battery: {battery_cap} mAh capacity. Fast Charging: {fast_charging_info}. "
            f"Cameras: Rear {num_rear_cam} (Primary {rear_cam_mp} MP), Front {num_front_cam} (Primary {front_cam_mp} MP). "
            f"Connectivity: 5G support: {'Yes' if has_5g else 'No'}, NFC support: {'Yes' if has_nfc else 'No'}. "
            f"Features: IR Blaster: {'Yes' if has_ir else 'No'}. Expandable Memory: {extended_memory_info}."
        )
        context_snippets.append(desc)

    logging.info(f"Generated {len(context_snippets)} individual context snippets.")
    print(f"Generated {len(context_snippets)} context snippets.")
    return context_snippets

def find_relevant_context(question, df, context_snippets):
    """
    Retrieves context snippet(s) relevant to the question using flexible keyword matching.
    Checks for brand and EITHER full model name OR base model name in the question.
    """
    if df is None or not context_snippets:
        logging.error("DataFrame or context snippets unavailable for retrieval.")
        return None, None  # Return None for context and matched models

    if 'base_model_lower' not in df.columns:
        logging.error("Required 'base_model_lower' column missing from DataFrame.")
        print("ERROR: Cannot perform flexible model matching, 'base_model_lower' column missing.")
        return None, None

    question_lower = question.lower()
    relevant_indices = []
    matched_models_set = set()

    for index, row in df.iterrows():
        brand_lower = str(row.get('brand_name', '')).lower()
        full_model_lower = str(row.get('model', '')).lower()
        base_model_lower = str(row.get('base_model_lower', '')).lower()

        if brand_lower:
            brand_match = re.search(r'\b' + re.escape(brand_lower) + r'\b', question_lower)

            if brand_match:
                base_model_match = False
                if base_model_lower:
                    if base_model_lower in question_lower:
                        base_model_match = True

                full_model_match = False
                if not base_model_match and full_model_lower:
                    if full_model_lower in question_lower:
                        full_model_match = True

                if base_model_match or full_model_match:
                    relevant_indices.append(index)
                    matched_models_set.add(f"{row.get('brand_name')} {row.get('model')}")
                    match_type = "base" if base_model_match else "full"
                    matched_name = base_model_lower if base_model_match else full_model_lower
                    logging.info(f"Match found for brand '{brand_lower}' using {match_type} model name '{matched_name}' in question.")

    relevant_indices = sorted(list(set(relevant_indices)))

    if not relevant_indices:
        logging.warning(f"Could not identify a specific phone model (brand & model/base model) in question: '{question}'")
        return None, None  # No specific context found
    else:
        focused_context_list = []
        for i in relevant_indices:
            if 0 <= i < len(context_snippets):
                focused_context_list.append(context_snippets[i])
            else:
                logging.warning(f"Retrieved index {i} out of bounds for snippets. Skipping.")

        if not focused_context_list:
            logging.error("No valid snippets found for the retrieved indices.")
            return None, None

        focused_context = " ".join(focused_context_list)
        matched_models_str = ", ".join(sorted(list(matched_models_set)))
        print(f"Retrieved context for model(s): {matched_models_str}")
        logging.info(f"Using focused context from {len(focused_context_list)} snippet(s) for '{matched_models_str}'.")
        return focused_context, matched_models_str

def load_qa_model(model_name):
    """Loads the specified RoBERTa tokenizer and model for Question Answering."""
    print(f"\nAttempting to load QA model: '{model_name}'...")
    tokenizer, model, device = None, None, None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using computational device: {device}")

        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        model = RobertaForQuestionAnswering.from_pretrained(model_name).to(device)
        model.eval()

        logging.info(f"Successfully loaded RoBERTa tokenizer and model '{model_name}' onto {device}.")
        print(f"Successfully loaded QA model '{model_name}' onto {device}.")
        return tokenizer, model, device
    except OSError as e:
        logging.warning(f"Could not find model '{model_name}' locally. Attempting download. Error: {e}")
        print(f"Model '{model_name}' not found locally. Downloading (this may take a while)...")
        try:
            tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            model = RobertaForQuestionAnswering.from_pretrained(model_name).to(device)
            model.eval()
            logging.info(f"Successfully downloaded and loaded model '{model_name}' onto {device}.")
            print(f"Successfully downloaded and loaded QA model '{model_name}' onto {device}.")
            return tokenizer, model, device
        except Exception as download_e:
            logging.exception(f"FATAL: Failed to download/load model '{model_name}'.")
            print(f"\n--- ERROR ---")
            print(f"Failed to download or load the QA model ('{model_name}').")
            print(f"Check your internet connection and the model name.")
            print(f"Detailed Error: {download_e}")
            print(f"-------------")
            return None, None, None
    except Exception as load_e:
        logging.exception(f"FATAL: An unexpected error occurred while loading the QA model '{model_name}'.")
        print(f"\n--- ERROR ---")
        print(f"An unexpected error occurred while loading the QA model ('{model_name}').")
        print(f"Detailed Error: {load_e}")
        print(f"-------------")
        return None, None, None

def answer_question_rag(question, df, context_snippets, tokenizer, model, device):
    """
    Answers a question using the RAG pipeline: Retrieve relevant context, then ask the QA model.
    """
    if tokenizer is None or model is None:
        return "ERROR: QA model is not loaded. Cannot answer questions."
    if not context_snippets:
        return "ERROR: Context snippets are not available. Cannot answer questions."

    focused_context, matched_models_str = find_relevant_context(question, df, context_snippets)

    if focused_context is None:
        return ("I couldn't identify a specific phone model (brand & model) mentioned in your question "
                "that matches the dataset closely enough. Please try asking about a specific model "
                "using both brand and model name (e.g., 'What is the RAM of the OnePlus 11 5G?').")

    try:
        inputs = tokenizer(question, focused_context,
                           add_special_tokens=True,
                           return_tensors="pt",
                           max_length=512,
                           truncation=True,
                           padding='max_length',
                           ).to(device)

        input_ids = inputs["input_ids"].squeeze()

        with torch.no_grad():
            outputs = model(**inputs)
            start_logits = outputs.start_logits.cpu()
            end_logits = outputs.end_logits.cpu()

        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        score_best_span = start_logits[0, start_index] + end_logits[0, end_index]
        score_cls = start_logits[0, 0] + end_logits[0, 0]

        is_unanswerable = False
        no_answer_reason = ""
        if start_index > end_index:
            is_unanswerable = True
            no_answer_reason = "Predicted start index is after end index."
        elif score_cls > score_best_span:
            is_unanswerable = True
            no_answer_reason = f"CLS token score ({score_cls:.2f}) is higher than best answer span score ({score_best_span:.2f})."
        elif start_index == 0 and end_index == 0:
            is_unanswerable = True
            no_answer_reason = "Predicted answer span is just the CLS token."

        if is_unanswerable:
            logging.warning(f"Model indicates no answer found for question '{question}'. Reason: {no_answer_reason}")
            if "CLS token score" in no_answer_reason:
                return f"Based on the information for {matched_models_str if matched_models_str else 'the relevant phone(s)'}, I couldn't find a specific answer to that part of your question in the text."
            else:
                return f"I found the data for {matched_models_str if matched_models_str else 'the relevant phone(s)'}, but couldn't extract a clear answer span. Could you try rephrasing?"

        if start_index < len(input_ids) and end_index < len(input_ids) and start_index <= end_index:
            answer_tokens = input_ids[start_index: end_index + 1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        else:
            logging.error(f"Predicted indices ({start_index}, {end_index}) out of bounds for input_ids length {len(input_ids)} or invalid range.")
            return "Error: Model predicted answer indices outside the text range."

        if not answer or answer.lower() in ['<s>', '</s>', '[cls]', '[sep]', '[pad]', '']:
            logging.warning(f"Decoded answer is empty or likely a special token: '{answer}'")
            return f"I found a potential match in the data for {matched_models_str if matched_models_str else 'the relevant phone(s)'}, but the extracted text was empty or unclear. Could you rephrase?"

        return answer

    except Exception as e:
        logging.exception(f"An error occurred during the QA step for question '{question}'")
        print(f"\n--- ERROR during QA processing ---")
        print(f"Question: {question}")
        print(f"Error: {e}")
        print(f"---------------------------------")
        return "Sorry, I encountered an error while trying to process your question."

print("--- Section 3: Function Definitions Complete ---")

# ---------------------------------------------------------------------------
# Section 4: Main Execution Logic
# ---------------------------------------------------------------------------
print("\n--- Section 4: Main Execution ---")

phone_df = load_data(CSV_FILE_PATH)

if phone_df is not None:
    print("\n--- Data Loaded and Cleaned Successfully ---")

    all_context_snippets = create_context_snippets(phone_df)

    if all_context_snippets:
        print("--- Context Snippets Created Successfully ---")

        tokenizer, model, device = load_qa_model(QA_MODEL_NAME)

        if tokenizer and model:
            print("\n--- QA Model Loaded Successfully ---")
            print("\n========================================================")
            print("=== Smartphone QA Bot (RAG approach) Initialized ===")
            print("========================================================")
            print(f"\nReady to answer questions about specific phones found in '{os.path.basename(CSV_FILE_PATH)}'.")
            print("Include both the BRAND and MODEL NAME in your question for best results.")
            print("Example: 'What is the battery capacity of the Samsung Galaxy A14?'")
            print("\nType 'quit' or 'exit' to stop.")

            while True:
                try:
                    question = input("\nYour Question: ")
                    question_stripped = question.strip()
                    question_lower = question_stripped.lower()

                    if question_lower in ['quit', 'exit']:
                        print("\nExiting QA Bot. Goodbye!")
                        break

                    if not question_stripped:
                        print("Please enter a question.")
                        continue

                    print("Processing...")
                    result = answer_question_rag(question_stripped, phone_df, all_context_snippets, tokenizer, model, device)

                    print(f"\nAnswer: {result}")

                except EOFError:
                    print("\nInput stream ended. Exiting.")
                    break
                except KeyboardInterrupt:
                    print("\nInterrupted by user. Exiting.")
                    break

        else:
            print("\n--- ERROR: Failed to load the QA model. Cannot start the interactive QA session. ---")

    else:
        print("\n--- ERROR: Failed to generate context snippets from the data. Cannot start QA Bot. ---")

else:
    print("\n--- FATAL ERROR: Failed to load data. QA pipeline cannot start. ---")
    print(f"--- Please ensure the file exists at '{CSV_FILE_PATH}' and is a valid, readable CSV file. ---")

print("\n--- QA Pipeline Finished ---")