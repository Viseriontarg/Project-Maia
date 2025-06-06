# ai_breakdown.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import re # Import regular expressions for cleaning

# --- Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# Configure the Generative AI library
genai.configure(api_key=GEMINI_API_KEY)

# --- Helper function to sanitize numeric parameters ---
def _sanitize_numeric_param(param_value):
    """
    Sanitizes a parameter value that is expected to be numeric.
    - Leaves existing numbers as is.
    - Converts valid number strings (e.g., "10.5") to float.
    - Converts "N/A" (or similar) to None.
    - Extracts number from strings like "15 days", "15 N/A", "$2000".
    - Returns None for any other non-numeric string or unexpected type.
    """
    if isinstance(param_value, (int, float)):
        return param_value

    if param_value is None:
        return None

    if isinstance(param_value, str):
        cleaned_str = param_value.strip()
        
        if cleaned_str.upper() in ['N/A', 'NA', 'NOT APPLICABLE']:
            return None

        # Regex to find the first valid number (integer or float) in the string
        match = re.match(r"^[^\d-]*(-?\d+(?:\.\d+)?)", cleaned_str)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return None
        else:
            return None # The string contains no parsable number at the start
    
    # Fallback for any other unexpected type
    return None


# --- The Core AI Function ---

def get_detailed_breakdown(simple_data: dict) -> dict:
    """
    Takes simple project data (name, type, cost, duration) and uses Gemini
    to generate a detailed project breakdown suitable for Monte Carlo simulation.
    """

    # --- 1. Define the Target JSON Structure (Helper for the Prompt) ---
    target_structure_example = """
    {
      "projectName": "...",
      "teamSize": 5, // AI-Generated (2-15)
      "dailyCost": 600, // AI-Generated (400-800)
      "simRuns": 10000,
      "phases": [
        {
          "phaseName": "Phase 1: ...",
          "tasks": [
            {
              "id": "T1", "name": "...", 
              "duration_params": {
                "type": "PERT", // "PERT", "Triangular", "Normal", "Uniform"
                "optimistic": 10, "most_likely": 15, "pessimistic": 25
              }, "predecessors": []
            }
          ]
        }
      ],
      "risks": [
        {
          "id": "R1", "name": "...", "probability": 0.15,
          "impactDays": {"type": "Triangular", "optimistic": 5, "most_likely": 10, "pessimistic": 15},
          "impactCost": {"type": "Normal", "mean": 10000, "std_dev": 2000},
          "correlates": {}
        }
      ]
    }
    """

    # --- 2. Build the Prompt for the Gemini API ---
    prompt = f"""
    You are an expert Project Manager AI. Your task is to expand high-level project information into a detailed, structured JSON plan suitable for Monte Carlo simulation.
    
    It is **ESSENTIAL** that the plan you generate aligns **closely** with the user's target cost and duration.

    High-Level Information:
    * Project Name: {simple_data.get('projectName', 'N/A')}
    * Project Type: {simple_data.get('projectType', 'N/A')}
    * Target Base Cost ($): {simple_data.get('baseCost', 'N/A')}
    * Target Base Duration (Days): {simple_data.get('baseDuration', 'N/A')}

    Your JSON Output MUST follow this structure:
    {target_structure_example}

    Instructions:
    1.  **Core Parameters:** Propose a plausible 'teamSize' (integer 2-15) and 'dailyCost' (integer 400-800).
    2.  **Phases & Tasks:** Create 3-5 phases with 3-5 tasks each. Assign unique 'id's and logical 'predecessors'.
    3.  **Task Durations:**
        - Choose a 'type': "PERT", "Triangular", "Normal", or "Uniform". Default to "PERT" for project tasks.
        - **CRITICAL:** Provide ONLY the parameters for the chosen type (e.g., for "PERT", only provide 'optimistic', 'most_likely', 'pessimistic'). DO NOT include parameters for other distributions (like 'mean' or 'min') in the same object.
        - All parameter values MUST be JSON numbers (e.g., 15), NOT strings (e.g., "15").
        - The expected duration (sum of 'most_likely' or 'mean' on the critical path) MUST align with the 'Target Base Duration'.
    4.  **Risks:** Create 3-5 relevant risks. Estimate 'probability' (number between 0.01-0.5).
    5.  **Risk Impacts:** For 'impactDays' and 'impactCost', choose a distribution 'type' and provide ONLY the relevant numeric parameters, just like with tasks.
    6.  **Validation:** Ensure the calculated likely cost and duration are plausible and close to the targets.
    7.  **Fixed Values:** Set 'simRuns' to 10000. Set 'correlates' to {{}}.
    8.  **Output Format:** Provide ONLY the valid JSON object. No extra text, no markdown, no explanations.

    Generate the JSON plan now.
    """

    # --- 3. Call the Gemini API ---
    print(">>> Calling Gemini API to generate project breakdown...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        response = model.generate_content(prompt)
        print(">>> Gemini API response received.")
        
        raw_text = response.text
        json_text = raw_text.strip()
        # Clean markdown code blocks if they exist
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()
        
        print(f">>> Attempting to parse JSON:\n{json_text[:500]}...")
        parsed_json = json.loads(json_text)
        print(">>> JSON parsed successfully!")

        # --- 4. Sanitize Parsed JSON Data ---
        print(">>> Sanitizing JSON data...")
        param_keys_to_sanitize = ["optimistic", "most_likely", "pessimistic", "mean", "std_dev", "min", "max"]

        # Sanitize root-level numeric fields
        for key in ["teamSize", "dailyCost", "simRuns"]:
            if key in parsed_json:
                parsed_json[key] = _sanitize_numeric_param(parsed_json[key])

        # Sanitize tasks
        if parsed_json.get("phases") and isinstance(parsed_json["phases"], list):
            for phase in parsed_json["phases"]:
                if phase.get("tasks") and isinstance(phase["tasks"], list):
                    for task in phase["tasks"]:
                        if task.get("duration_params") and isinstance(task["duration_params"], dict):
                            for p_key in param_keys_to_sanitize:
                                if p_key in task["duration_params"]:
                                    task["duration_params"][p_key] = _sanitize_numeric_param(task["duration_params"][p_key])
        
        # Sanitize risks
        if parsed_json.get("risks") and isinstance(parsed_json["risks"], list):
            for risk in parsed_json["risks"]:
                if "probability" in risk:
                    risk["probability"] = _sanitize_numeric_param(risk["probability"])
                
                for impact_category in ["impactDays", "impactCost"]:
                    if risk.get(impact_category) and isinstance(risk.get(impact_category), dict):
                        for p_key in param_keys_to_sanitize:
                            if p_key in risk[impact_category]:
                                risk[impact_category][p_key] = _sanitize_numeric_param(risk[impact_category][p_key])
        
        print(">>> JSON data sanitized.")
        
        return parsed_json
        
    except json.JSONDecodeError as e:
        print(f"!!! FATAL ERROR: Failed to parse Gemini response as JSON. Error: {e} !!!")
        print(f"!!! Raw Response was:\n{raw_text} !!!")
        raise ValueError(f"AI response was not valid JSON. Please check the prompt or AI model. Error: {e}")
    except Exception as e:
        print(f"!!! FATAL ERROR: An error occurred during Gemini API call or processing: {e} !!!")
        raise e

# --- Example Usage ---
if __name__ == '__main__':
    # This block allows you to test the AI breakdown function directly.
    # NOTE: You must have a .env file with a valid GEMINI_API_KEY for this to run.
    print("--- Running AI Breakdown Test ---")
    
    # 1. Define some simple, high-level project data
    sample_project_data = {
        'projectName': 'New E-commerce Platform Launch',
        'projectType': 'Web Development',
        'baseCost': 250000,
        'baseDuration': 120
    }
    
    print(f"\n[Step 1] Using sample data:\n{json.dumps(sample_project_data, indent=2)}")

    # 2. Call the function to get the detailed breakdown from the AI
    try:
        detailed_plan = get_detailed_breakdown(sample_project_data)
        
        # 3. Print the resulting detailed plan
        print("\n[Step 2] Successfully generated detailed plan:")
        print(json.dumps(detailed_plan, indent=2))
        
        # You could now save this to a file, e.g., with open('output.json', 'w') as f: ...
        print("\n--- AI Breakdown Test Complete ---")

    except ValueError as e:
        print(f"\n--- AI Breakdown Test Failed ---")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n--- AI Breakdown Test Failed with an unexpected error ---")
        print(f"Error: {e}")