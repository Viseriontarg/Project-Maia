# Maia: AI-Powered Project Planning and Simulation

## Project Title and Introduction

**BusinessDigitalTwin** is a powerful project management tool that leverages generative AI and Monte Carlo simulations to transform high-level project goals into actionable plans and realistic forecasts. It helps users understand potential project durations and costs by considering inherent uncertainties and risks, ultimately enabling more informed decision-making.

## Features

-   **AI-Powered Project Planning**: Utilizes Google's Gemini generative AI to break down broad project objectives into detailed tasks, phases, and potential risks. Each task is assigned parameters for duration (optimistic, most likely, pessimistic) and each risk is given probability and impact parameters.
-   **Monte Carlo Simulation**: Employs Monte Carlo methods to simulate thousands of possible project outcomes based on the AI-generated plan. This provides a probabilistic view of project duration and cost.
-   **Excel Report Generation**: Outputs comprehensive simulation results into a structured and easy-to-understand Excel file. This report includes P10, P50, and P90 estimates for project duration and cost, along with risk driver analysis.

## How it Works

1.  **User Input**: The user provides initial project data through a web interface. This includes the project name, project type (e.g., "Software Development", "Marketing Campaign"), target cost, and target duration.
2.  **AI-Driven Plan Generation**: The backend (`ai_breakdown.py`) sends this high-level data to the Gemini API. The AI then generates a detailed project plan, structured as a JSON object. This plan includes:
    *   Phases of the project.
    *   Tasks within each phase, with estimated optimistic, most likely, and pessimistic durations.
    *   Potential risks, each with an estimated probability of occurrence and impact on duration and/or cost.
3.  **Monte Carlo Simulation**: The detailed JSON plan is passed to the simulation module (`simulation.py`). This module runs a Monte Carlo simulation, performing many iterations (e.g., 10,000+) to model the range of potential project outcomes based on the PERT distribution for task durations and the defined risks.
4.  **Report Generation**: The results from the simulation, including statistical measures like P10 (optimistic), P50 (most likely), and P90 (pessimistic) for both total project duration and cost, are compiled. The `report_generator.py` module then creates a formatted Excel report summarizing these findings and highlighting key risk drivers.
5.  **Output**: The user receives a confirmation, and the Excel report is saved to the server, typically in an `outputs` directory.

## Project Structure

-   `app.py`: The main Flask web application. It handles HTTP routing, serves the user interface, and orchestrates the calls to the AI breakdown, simulation, and report generation modules.
-   `ai_breakdown.py`: Contains the logic for interacting with the Google Gemini API. It takes high-level project information and prompts the AI to produce a detailed, structured JSON project plan with tasks, durations (optimistic, most-likely, pessimistic), and risks (probability, impact).
-   `simulation.py`: Performs the Monte Carlo simulation. It takes the detailed project plan (JSON) as input, simulates project execution many times using PERT distributions for task durations and incorporating specified risks, and calculates overall project duration and cost distributions.
-   `report_generator.py`: Takes the raw simulation results and the initial project input data to create a well-formatted Excel report. This report includes summary statistics (P10, P50, P90 for cost and duration), and often visualizations or tables showing risk impacts.
-   `templates/decisonscope-landing.html`: The HTML frontend that provides the user interface for inputting initial project details.
-   `outputs/`: (Typically created by the application) Directory where generated Excel reports are stored.
-   `.env`: (User-created) File to store sensitive credentials like the API key.

## Setup and Running the Project

### Prerequisites

-   Python 3.7+
-   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd BusinessDigitalTwin
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    It's recommended to have a `requirements.txt` file. If one is not available, install the packages directly:
    ```bash
    pip install Flask google-generativeai pandas numpy openpyxl python-dotenv scipy
    ```
    *(If a `requirements.txt` file is present in the repository, use `pip install -r requirements.txt`)*

4.  **API Key Setup:**
    Create a `.env` file in the root directory of the project:
    ```
    GEMINI_API_KEY='YOUR_GEMINI_API_KEY_HERE'
    ```
    Replace `YOUR_GEMINI_API_KEY_HERE` with your actual Google Gemini API key.

### Running the Application

1.  **Ensure your virtual environment is activated** (if you created one).
2.  **Run the Flask application:**
    ```bash
    python app.py
    ```
3.  The application will start, and you can access it in your web browser at `http://127.0.0.1:5000/`.

## API Endpoints

The application provides the following API endpoints:

-   **`GET /`**:
    -   Description: Serves the main landing page (`decisonscope-landing.html`) where users can input project details.
    -   Response: HTML content of the landing page.

-   **`POST /generate-plan`**:
    -   Description: Receives initial project data from the web form. It then calls the Gemini AI to break down the project into a detailed plan.
    -   Request Body (JSON):
        ```json
        {
            "project_name": "New Marketing Campaign",
            "project_type": "Marketing",
            "target_cost": 100000,
            "target_duration": 90 // in days
        }
        ```
    -   Response (JSON): The detailed project plan generated by the AI, including phases, tasks (with optimistic, most_likely, pessimistic durations), and risks (with probability and impact).
        ```json
        {
          "project_name": "New Marketing Campaign",
          "phases": [
            {
              "phase_name": "Phase 1: Research & Strategy",
              "tasks": [
                {
                  "task_name": "Market Research",
                  "optimistic_duration": 5,
                  "most_likely_duration": 7,
                  "pessimistic_duration": 10
                },
                // ... more tasks ...
              ]
            }
            // ... more phases ...
          ],
          "risks": [
            {
              "risk_name": "Key Competitor Launch",
              "probability": 0.2, // 20%
              "cost_impact_percentage": 10, // 10% increase in cost
              "duration_impact_days": 7 // 7 days delay
            }
            // ... more risks ...
          ]
        }
        ```

-   **`POST /simulate`**:
    -   Description: Receives the detailed project plan (typically from the `/generate-plan` step or a stored plan). It runs the Monte Carlo simulation and then generates an Excel report.
    -   Request Body (JSON): The detailed project plan structure as returned by `/generate-plan`.
    -   Response (JSON): A success message indicating the path to the generated Excel report.
        ```json
        {
            "message": "Simulation complete. Report generated at outputs/New_Marketing_Campaign_simulation_report.xlsx",
            "report_path": "outputs/New_Marketing_Campaign_simulation_report.xlsx"
        }
        ```
        (The exact filename will vary based on the project name and timestamp).
