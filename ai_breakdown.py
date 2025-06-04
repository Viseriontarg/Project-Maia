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

# --- The Core AI Function ---

def get_detailed_breakdown(simple_data: dict) -> dict:
    """
    Takes simple project data (name, type, cost, duration) and uses Gemini
    to generate a detailed project breakdown suitable for Monte Carlo simulation.

    Args:
        simple_data: A dictionary containing projectName, projectType,
                     baseCost, and baseDuration.

    Returns:
        A dictionary with the detailed project structure.

    Raises:
        ValueError: If the AI response cannot be parsed as valid JSON.
        Exception: For API call errors or other issues.
    """

    # --- 1. Define the Target JSON Structure (Helper for the Prompt) ---
    # This helps the AI understand exactly what we need.
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
              "id": "T1", 
              "name": "...", 
              "duration_params": {
                "type": "PERT", // Options: "PERT", "Triangular", "Normal", "Uniform"
                // For PERT or Triangular:
                "optimistic": 10, 
                "most_likely": 15, 
                "pessimistic": 25
                // For Normal (use instead of above if type is "Normal"):
                // "mean": 15, "std_dev": 3
                // For Uniform (use instead of above if type is "Uniform"):
                // "min": 10, "max": 20
              },
              "predecessors": []
            },
            {
              "id": "T2", 
              "name": "...", 
              "duration_params": {
                "type": "Normal",
                "mean": 20,
                "std_dev": 4
              },
              "predecessors": ["T1"]
            }
          ]
        }
      ],
      "risks": [
        {
          "id": "R1",
          "name": "...",
          "probability": 0.15,
          "impactDays": {
            "type": "Triangular",
            "optimistic": 5,
            "most_likely": 10,
            "pessimistic": 15
          },
          "impactCost": {
            "type": "Normal",
            "mean": 10000,
            "std_dev": 2000
          },
          "correlates": {}
        }
      ]
    }
    """

    # --- 2. Build the Prompt for the Gemini API ---
    prompt = f"""
    You are an expert Project Manager AI with deep knowledge of statistical distributions. Your task is to expand high-level project information into a detailed, structured JSON plan suitable for Monte Carlo simulation using appropriate statistical distributions.
    
    It is **ESSENTIAL** that the plan you generate aligns **closely** with the user's target cost and duration.

    High-Level Information:
    * Project Name: {simple_data.get('projectName', 'N/A')}
    * Project Type: {simple_data.get('projectType', 'N/A')}
    * Target Base Cost ($): {simple_data.get('baseCost', 'N/A')}
    * Target Base Duration (Days): {simple_data.get('baseDuration', 'N/A')}

    Your JSON Output MUST follow this structure:
    {target_structure_example}

    Instructions:
    1. **Determine Core Parameters:** Based on the Project Type and Targets, propose a plausible 'teamSize' (integer between 2 and 15) and 'dailyCost' (integer between 400 and 800).
    
    2. **Generate Phases & Tasks:** Create 3-5 phases with 3-5 tasks each, relevant to the 'Project Type'. Assign unique 'id's (T1, T2...) and logical 'predecessors'.
    
    3. **Select Distribution Types for Task Durations:**
       - For project management tasks (Software Development, Construction, Infrastructure Rollout), default to "PERT" distribution as it's well-suited for time estimates
       - Use "Triangular" for tasks with less certain estimates or simpler scenarios
       - Use "Normal" for well-understood, repetitive tasks with predictable variation
       - Use "Uniform" only when any duration within a range is equally likely (rare for tasks)
    
    4. **Estimate Task Duration Parameters:** 
       - For PERT/Triangular: Assign 'optimistic', 'most_likely', 'pessimistic' values
       - For Normal: Assign 'mean' and 'std_dev' (standard deviation should be ~15-20% of mean for typical tasks)
       - For Uniform: Assign 'min' and 'max'
       - **CRITICAL:** The expected values (most_likely for PERT/Triangular, mean for Normal, midpoint for Uniform) along the critical path MUST closely match the 'Target Base Duration'
    
    5. **Generate Risks:** Create 3-5 relevant risks. Assign 'id's (R1, R2...). Estimate 'probability' (0.01-0.5).
    
    6. **Select Distribution Types for Risk Impacts:**
       - Use "Normal" if impacts cluster around an average value
       - Use "PERT" or "Triangular" when you can estimate optimistic/most_likely/pessimistic scenarios
       - Use "Uniform" if any impact within a range is equally likely
    
    7. **Estimate Risk Impact Parameters:**
       - Assign appropriate parameters based on the chosen distribution type
       - Ensure parameters are realistic for the risk type and project context
    
    8. **Validate Cost:** Calculate estimated 'likely' cost: (Total Expected Duration * teamSize * dailyCost) + (Sum of (risk_probability * risk_expected_impact_cost))
       - Expected duration = sum of most_likely/mean/midpoint values along critical path
       - Expected impact cost = most_likely/mean/midpoint of each risk's cost impact
       - This MUST closely match the 'Target Base Cost'. Adjust parameters while maintaining plausibility.
    
    9. **Set Fixed Values:** Set 'simRuns' to 10000. Set 'correlates' to {{}}.
    
    10. **Output Format:** Provide ONLY the valid JSON object. No extra text, no markdown. Include only the parameters relevant to each chosen distribution type.

    Generate the JSON plan now.
    """

    # --- 3. Call the Gemini API ---
    print(">>> Calling Gemini API to generate project breakdown...")
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # Ensure this is the working model name
        response = model.generate_content(prompt)
        print(">>> Gemini API response received.")
        
        # --- 4. Clean and Parse the Response ---
        # Sometimes the AI might still wrap the JSON in markdown or add extra text.
        # We'll try to clean it, but a good prompt is the best defense.
        raw_text = response.text
        
        # Use regex to find the JSON block (handles optional ```json ... ```)
        match = re.search(r'```json\s*([\s\S]*?)\s*```|([\s\S]*)', raw_text)
        if match:
            json_text = match.group(1) if match.group(1) else match.group(2)
            json_text = json_text.strip() # Remove leading/trailing whitespace
        else:
             json_text = raw_text.strip()

        print(f">>> Attempting to parse JSON:\n{json_text[:500]}...") # Print first 500 chars

        parsed_json = json.loads(json_text)
        print(">>> JSON parsed successfully!")
        return parsed_json

    except json.JSONDecodeError as e:
        print(f"!!! FATAL ERROR: Failed to parse Gemini response as JSON. Error: {e} !!!")
        print(f"!!! Raw Response was:\n{raw_text} !!!")
        raise ValueError(f"AI response was not valid JSON. Please check the prompt or AI model. Error: {e}")
    except Exception as e:
        print(f"!!! FATAL ERROR: An error occurred during Gemini API call or processing: {e} !!!")
        # You might want to check response.prompt_feedback here for safety ratings
        # print(f"Prompt Feedback: {response.prompt_feedback}")
        raise e

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    # This part runs only when you execute `python ai_breakdown.py` directly
    test_input = {
        "projectName": "New Mobile App Launch",
        "projectType": "Software Development",
        "baseCost": 150000,
        "baseDuration": 90
    }
    try:
        detailed_plan = get_detailed_breakdown(test_input)
        # Pretty-print the output JSON
        print("\n--- Generated Detailed Plan ---")
        print(json.dumps(detailed_plan, indent=4))
        print("\n--- AI Breakdown Module Test Successful ---")
    except Exception as e:
        print(f"\n--- AI Breakdown Module Test Failed: {e} ---")