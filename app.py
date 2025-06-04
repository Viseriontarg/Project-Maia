# app.py

import os
from flask import Flask, render_template, request, jsonify # request and jsonify are key for this update
from simulation import calculate_scenarios # Assuming simulation.py is in the same directory or Python path
from report_generator import create_excel_report # Assuming report_generator.py is in the same directory or Python path
from flask_cors import CORS # For handling CORS if needed
import traceback # For detailed error logging
from ai_breakdown import get_detailed_breakdown # Added as per instructions

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

# --- Constants for Validation ---
# Define valid project types as per decisonscope-landing.html
VALID_PROJECT_TYPES = {
    "Software Development",
    "Infrastructure Rollout",
    "Marketing Campaign",
    "Construction Project",
    "Research & Development",
    "Business Process Change"
}

# Maximum length for project name
MAX_PROJECT_NAME_LENGTH = 100

# --- Routes (Endpoints) ---

@app.route('/', methods=['GET'])
def landing_page():
    """
    Serves the HTML landing page.
    Assumes 'decisonscope-landing.html' is in a 'templates' folder
    in the same directory as app.py.
    """
    return render_template('decisonscope-landing.html')

@app.route('/simulate', methods=['GET', 'POST']) # Allow GET for info, POST for simulation
def run_simulation():
    if request.method == 'POST':
        print("\n--- /simulate endpoint hit (POST) ---")
        try:
            # 1. Get JSON data from the request
            try:
                frontend_data = request.get_json()
                if frontend_data is None:
                    print("\n!!! Error: No JSON data found in request or JSON body is null. !!!\n")
                    return jsonify({'message': 'Error: Invalid JSON or Content-Type header. Ensure JSON payload is not null.'}), 400
            except Exception as json_parsing_error:
                print(f"\n!!! Error parsing JSON or incorrect Content-Type: {json_parsing_error} !!!\n")
                traceback.print_exc()
                return jsonify({'message': 'Error: Invalid JSON or Content-Type header.'}), 400

            print(f"\n>>> Received JSON data from frontend: {frontend_data}")

            # 2. Comprehensive Data Validation
            # Create a list to collect all validation errors
            validation_errors = []
            
            # 2.1 Validate projectName
            project_name_from_frontend = frontend_data.get('projectName')
            
            # Check if projectName exists and is a string
            if project_name_from_frontend is None:
                validation_errors.append("projectName is required.")
            elif not isinstance(project_name_from_frontend, str):
                validation_errors.append("projectName must be a string.")
            elif not project_name_from_frontend.strip():  # Check for empty or whitespace-only
                validation_errors.append("projectName must not be empty or consist only of whitespace.")
            elif len(project_name_from_frontend) > MAX_PROJECT_NAME_LENGTH:
                validation_errors.append(f"projectName must not exceed {MAX_PROJECT_NAME_LENGTH} characters.")
            
            # 2.2 Validate projectType
            project_type_from_frontend = frontend_data.get('projectType')
            
            # Check if projectType exists and is a string
            if project_type_from_frontend is None:
                validation_errors.append("projectType is required.")
            elif not isinstance(project_type_from_frontend, str):
                validation_errors.append("projectType must be a string.")
            elif not project_type_from_frontend.strip():  # Check for empty string
                validation_errors.append("projectType must not be empty.")
            elif project_type_from_frontend not in VALID_PROJECT_TYPES:
                validation_errors.append(
                    f"projectType must be one of: {', '.join(sorted(VALID_PROJECT_TYPES))}."
                )
            
            # 2.3 Validate baseCost (optional field)
            base_cost_from_frontend = frontend_data.get('baseCost')
            
            # baseCost can be null, but if provided, must be a non-negative number
            if base_cost_from_frontend is not None:
                # Check if it's a number (int or float)
                if not isinstance(base_cost_from_frontend, (int, float)):
                    validation_errors.append("baseCost must be a number if provided.")
                elif base_cost_from_frontend < 0:
                    validation_errors.append("baseCost must be a non-negative number.")
                # Additional check for NaN or infinity
                elif not isinstance(base_cost_from_frontend, bool) and (
                    base_cost_from_frontend != base_cost_from_frontend or  # NaN check
                    base_cost_from_frontend == float('inf') or 
                    base_cost_from_frontend == float('-inf')
                ):
                    validation_errors.append("baseCost must be a valid finite number.")
            
            # 2.4 Validate baseDuration (optional field)
            base_duration_from_frontend = frontend_data.get('baseDuration')
            
            # baseDuration can be null, but if provided, must be a positive integer
            if base_duration_from_frontend is not None:
                # Check if it's a number first
                if not isinstance(base_duration_from_frontend, (int, float)):
                    validation_errors.append("baseDuration must be a number if provided.")
                # If it's a float, check if it's a whole number
                elif isinstance(base_duration_from_frontend, float):
                    if base_duration_from_frontend.is_integer():
                        # Convert to int for further validation
                        base_duration_from_frontend = int(base_duration_from_frontend)
                        frontend_data['baseDuration'] = base_duration_from_frontend
                    else:
                        validation_errors.append("baseDuration must be a whole number (integer).")
                # Now check if it's positive (only if we haven't already found it's not an integer)
                elif isinstance(base_duration_from_frontend, int) and base_duration_from_frontend < 1:
                    validation_errors.append("baseDuration must be a positive integer (>= 1).")
            
            # 2.5 If there are validation errors, return them
            if validation_errors:
                # Join all errors into a single message
                error_message = "Validation Error: " + " ".join(validation_errors)
                print(f"\n!!! Validation failed: {error_message} !!!\n")
                return jsonify({'message': error_message}), 400
            
            # If we reach here, all validation passed
            print("\n>>> All validation checks passed.")
            
            # Log received project details before AI call
            print(f"\n>>> Frontend Project Name: {project_name_from_frontend}")
            print(f"    Frontend Project Type: {project_type_from_frontend}")
            if base_cost_from_frontend is not None:
                print(f"    Frontend Base Cost: {base_cost_from_frontend}")
            if base_duration_from_frontend is not None:
                print(f"    Frontend Base Duration: {base_duration_from_frontend}")

            # 3. Call AI module to generate detailed project plan
            # This section replaces the old sample_input_data manipulation
            print("\n>>> Calling get_detailed_breakdown...")
            try:
                # Call the AI module to generate the detailed plan
                detailed_input_data = get_detailed_breakdown(frontend_data)
                print(">>> AI breakdown successful.")
                if detailed_input_data and 'projectName' in detailed_input_data:
                    print(f"    AI Generated Project Name: {detailed_input_data.get('projectName')}")
                else:
                    # This case might indicate an issue with the AI's output structure
                    print("    AI breakdown returned data, but 'projectName' might be missing or data is None/unexpected.")
                    if detailed_input_data is None: # Specifically if AI returns None
                        # Initialize to empty dict to allow adding target values
                        detailed_input_data = {}
                        print("    Warning: detailed_input_data from AI was None. Initializing to empty dict to preserve user targets.")

                # Add original user targets for reporting purposes
                # These keys ('targetBaseCost', 'targetBaseDuration') are what report_generator.py now expects
                detailed_input_data['targetBaseCost'] = base_cost_from_frontend
                detailed_input_data['targetBaseDuration'] = base_duration_from_frontend
                print(f"    Added user targets - targetBaseCost: {detailed_input_data['targetBaseCost']}, targetBaseDuration: {detailed_input_data['targetBaseDuration']}")

            except Exception as ai_error:
                print(f"\n!!! An error occurred calling get_detailed_breakdown: {ai_error} !!!\n")
                traceback.print_exc() # Ensure 'import traceback' is present
                # Return a specific error message to the frontend
                return jsonify({'message': f'Error generating project plan via AI: {str(ai_error)}'}), 500

            # 4. Call Simulation
            print("\n>>> Calling calculate_scenarios...")
            sim_results = calculate_scenarios(detailed_input_data) # Use detailed_input_data
            print("\n>>> Simulation Results Received (summary or type):")
            if isinstance(sim_results, dict) and 'summary_stats' in sim_results:
                 print(f"    Summary Stats (Duration P10,P50,P90): {sim_results['summary_stats'].get('duration_p10_p50_p90_days', 'N/A')}")
                 print(f"    Summary Stats (Cost P10,P50,P90): {sim_results['summary_stats'].get('cost_p10_p50_p90', 'N/A')}")
            else:
                 print(f"    Simulation results type: {type(sim_results)}")

            # 5. Call Report Generator
            print("\n>>> Calling create_excel_report...")
            file_path = create_excel_report(detailed_input_data, sim_results) # Use detailed_input_data
            print(f"\n>>> Report generated at: {file_path}")

            # 6. Create Success Response
            print("\n--- POST Request successful ---\n")
            return jsonify({'message': 'Success', 'report_path': file_path}), 200

        except Exception as e:
            print(f"\n!!! An error occurred during POST processing: {e} !!!\n")
            traceback.print_exc()
            return jsonify({'message': f'Error during simulation or report generation: {str(e)}'}), 500

    elif request.method == 'GET':
        print("\n--- /simulate endpoint hit (GET) ---")
        return jsonify({'message': 'Please use POST method with JSON data to run a simulation.'}), 405
    
    else:
        print(f"\n--- /simulate endpoint hit with unsupported method: {request.method} ---")
        return jsonify({'message': f'Method {request.method} not allowed.'}), 405
    
# --- Run the App ---
if __name__ == '__main__':
    if not os.path.exists("templates"):
        os.makedirs("templates")
    if not os.path.exists("templates/decisonscope-landing.html"):
        with open("templates/decisonscope-landing.html", "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DecisionScope Landing</title>
</head>
<body>
    <h1>Welcome to DecisionScope Simulation</h1>
    <p>Click the button to run a sample simulation (using data from this page if available, or defaults) and generate an Excel report.</p>
    
    <label for="projectName">Project Name:</label>
    <input type="text" id="projectName" name="projectName" value="User Input Project"><br><br>

    <label for="projectType">Project Type:</label> <!-- ADDED projectType input -->
    <input type="text" id="projectType" name="projectType" value="Software Development"><br><br>
    
    <!-- baseCost and baseDuration are not used by the backend yet, but included for future use -->
    <label for="baseCost">Base Cost (Optional):</label>
    <input type="number" id="baseCost" name="baseCost" value="150000"><br><br>
    
    <label for="baseDuration">Base Duration (Optional):</label>
    <input type="number" id="baseDuration" name="baseDuration" value="120"><br><br>

    <button id="runSimButton">Run Simulation with Input</button>
    
    <div id="responseMessage" style="margin-top: 20px;"></div>

    <script>
        const simButton = document.getElementById('runSimButton');
        const responseDiv = document.getElementById('responseMessage');
        const projectNameInput = document.getElementById('projectName');
        const projectTypeInput = document.getElementById('projectType'); // ADDED projectType input ref
        const baseCostInput = document.getElementById('baseCost');
        const baseDurationInput = document.getElementById('baseDuration');

        simButton.addEventListener('click', async function(event) {
            event.preventDefault();
            responseDiv.textContent = 'Processing... please wait.';

            const payload = {
                projectName: projectNameInput.value || "Default Frontend Project", 
                projectType: projectTypeInput.value || "Default Type", // ADDED projectType to payload
                baseCost: parseInt(baseCostInput.value) || null,
                baseDuration: parseInt(baseDurationInput.value) || null
            };
            
            // Ensure nulls are sent if inputs are empty, or remove if truly optional for AI
            if (baseCostInput.value === "") payload.baseCost = null;
            if (baseDurationInput.value === "") payload.baseDuration = null;


            try {
                const response = await fetch('/simulate', { 
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const data = await response.json(); 

                if (response.ok) { 
                    responseDiv.textContent = 'Message: ' + data.message;
                    if (data.report_path) {
                         responseDiv.innerHTML += '<br>Report Path: ' + data.report_path;
                         responseDiv.innerHTML += '<br>You can find the report in the application directory (or specified path).';
                    }
                } else { 
                    responseDiv.textContent = 'Error (HTTP ' + response.status + '): ' + (data.message || 'Unknown server error');
                }
            } catch (error) {
                console.error('Fetch error:', error);
                responseDiv.textContent = 'Network error or server unavailable: ' + error.toString();
            }
        });
    </script>
</body>
</html>""")
    # Ensure dummy files for simulation and report_generator exist for basic app run
    # ai_breakdown.py is assumed to exist as per problem description
    if not os.path.exists("simulation.py"):
        with open("simulation.py", "w") as f:
            f.write("""
# Dummy simulation.py
def calculate_scenarios(input_data):
    project_name = "Unknown Project"
    if input_data and isinstance(input_data, dict):
        project_name = input_data.get('projectName', 'Unknown Project')
    print(f"[simulation.py] Calculating scenarios for: {project_name}")
    # Simplified mock results
    sim_runs = 1000
    if input_data and isinstance(input_data, dict):
        sim_runs = input_data.get("simRuns", 1000)

    return {
        "summary_stats": {
            "duration_p10_p50_p90_days": [100, 120, 150],
            "cost_p10_p50_p90": [100000, 120000, 150000]
        },
        "all_runs_data": {
            "durations": [100, 110, 120, 120, 130, 150] * (sim_runs // 6 + 1),
            "costs": [100000, 110000, 120000, 120000, 130000, 150000] * (sim_runs // 6 + 1)
        },
        "risk_impact_summary": {} # Mocked
    }
""")
    if not os.path.exists("report_generator.py"):
        with open("report_generator.py", "w") as f:
            f.write("""
# Dummy report_generator.py
import os
def create_excel_report(input_data, sim_results):
    project_name = "UnnamedProject"
    if input_data and isinstance(input_data, dict):
        project_name = input_data.get("projectName", "UnnamedProject").replace(" ", "_")
    
    file_name = f"{project_name}_simulation_report.txt" # Using .txt for simplicity
    output_dir = "generated_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w") as report_file:
        report_file.write(f"Simulation Report for: {input_data.get('projectName', 'N/A') if input_data else 'N/A'}\\n")
        report_file.write("This is a dummy report.\\n")
        if sim_results and sim_results.get('summary_stats'):
            report_file.write(f"Duration P50: {sim_results['summary_stats']['duration_p10_p50_p90_days'][1]} days\\n")
            report_file.write(f"Cost P50: {sim_results['summary_stats']['cost_p10_p50_p90'][1]}\\n")
        else:
            report_file.write("Simulation results are missing or incomplete.\\n")

    print(f"[report_generator.py] Dummy report created at: {file_path}")
    return file_path
""")
    app.run(debug=True, port=5000)