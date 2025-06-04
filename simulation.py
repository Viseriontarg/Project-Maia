import numpy as np
import random
import copy
import json # For pretty printing in __main__
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import norm, uniform, triang, beta

# -------------------- PERT Distribution Implementation --------------------

def pert_rvs(a, b, c, size=None):
    """
    Generate random variates from a PERT distribution.
    
    Parameters:
    a: minimum value (optimistic)
    b: most likely value (mode)
    c: maximum value (pessimistic)
    size: output shape (optional)
    
    Returns:
    Random variates from PERT distribution
    """
    if a >= c:
        return a if size is None else np.full(size, a)
    
    # PERT distribution parameters
    # Alpha and Beta parameters for the underlying Beta distribution
    mean = (a + 4*b + c) / 6
    
    if a == c:
        return a if size is None else np.full(size, a)
    
    # Standard PERT formulation
    alpha = 1 + 4 * (b - a) / (c - a)
    beta_param = 1 + 4 * (c - b) / (c - a)
    
    # Ensure alpha and beta are positive
    alpha = max(alpha, 0.1)
    beta_param = max(beta_param, 0.1)
    
    # Generate beta random variates and scale to [a, c]
    beta_samples = beta.rvs(alpha, beta_param, size=size)
    return a + (c - a) * beta_samples

# -------------------- Helper Functions --------------------

def _sample_from_distribution(dist_params: Dict[str, Any]) -> float:
    """
    Samples from a specified distribution based on the distribution type and parameters.
    
    Args:
        dist_params: Dictionary containing 'type' and distribution-specific parameters
        
    Returns:
        A single sample from the specified distribution
        
    Raises:
        ValueError: If distribution type is unknown or parameters are missing
    """
    dist_type = dist_params.get('type', '').upper()
    
    if dist_type == 'PERT':
        # PERT distribution parameters
        optimistic = float(dist_params.get('optimistic', 0))
        most_likely = float(dist_params.get('most_likely', 0))
        pessimistic = float(dist_params.get('pessimistic', 0))
        
        # Handle edge case where all values are the same
        if optimistic == pessimistic:
            return optimistic
            
        # Use our custom PERT implementation
        return pert_rvs(optimistic, most_likely, pessimistic)
            
    elif dist_type == 'NORMAL':
        # Normal distribution parameters
        mean = float(dist_params.get('mean', 0))
        std_dev = float(dist_params.get('std_dev', 0))
        
        # Handle edge case where std_dev is 0
        if std_dev <= 0:
            return mean
            
        return norm.rvs(loc=mean, scale=std_dev)
        
    elif dist_type == 'UNIFORM':
        # Uniform distribution parameters
        min_val = float(dist_params.get('min', 0))
        max_val = float(dist_params.get('max', 0))
        
        # Handle edge case where min equals max
        if min_val >= max_val:
            return min_val
            
        return uniform.rvs(loc=min_val, scale=max_val - min_val)
        
    elif dist_type == 'TRIANGULAR':
        # Triangular distribution parameters
        optimistic = float(dist_params.get('optimistic', 0))
        most_likely = float(dist_params.get('most_likely', 0))
        pessimistic = float(dist_params.get('pessimistic', 0))
        
        # Handle edge case where all values are the same
        if optimistic == pessimistic:
            return optimistic
            
        # Map to scipy.stats.triang parameters
        loc = optimistic
        scale = pessimistic - optimistic
        if scale > 0:
            c = (most_likely - optimistic) / scale
            # Ensure c is within valid range [0, 1]
            c = max(0, min(1, c))
            return triang.rvs(c, loc=loc, scale=scale)
        else:
            return optimistic
            
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def _get_triangular_sample(best: float, likely: float, worst: float) -> float:
    """
    Legacy function for backward compatibility.
    Generates a random sample from a triangular distribution.
    """
    # Create a distribution params dict and use the new function
    dist_params = {
        'type': 'TRIANGULAR',
        'optimistic': best,
        'most_likely': likely,
        'pessimistic': worst
    }
    return _sample_from_distribution(dist_params)

def _calculate_task_duration_sample(task_details: Dict[str, Any]) -> float:
    """Calculates a single task duration sample based on the specified distribution."""
    # Check for new structure with duration_params
    if 'duration_params' in task_details:
        return _sample_from_distribution(task_details['duration_params'])
    
    # Fallback to old structure for backward compatibility
    elif all(key in task_details for key in ['best', 'likely', 'worst']):
        return _get_triangular_sample(
            task_details['best'],
            task_details['likely'],
            task_details['worst']
        )
    else:
        raise ValueError("Task details must contain either 'duration_params' or 'best/likely/worst' values")

def _calculate_risk_impact_sample(impact_details: Dict[str, Any]) -> float:
    """Calculates a single risk impact (days or cost) sample based on the specified distribution."""
    # Check if impact_details has a 'type' field (new structure)
    if 'type' in impact_details:
        return _sample_from_distribution(impact_details)
    
    # Fallback to old structure for backward compatibility
    elif all(key in impact_details for key in ['best', 'likely', 'worst']):
        return _get_triangular_sample(
            impact_details['best'],
            impact_details['likely'],
            impact_details['worst']
        )
    else:
        raise ValueError("Impact details must contain either a distribution 'type' or 'best/likely/worst' values")

def _calculate_phase_critical_path_duration(
    tasks_in_phase: List[Dict[str, Any]],
    task_durations_this_run: Dict[str, float]
) -> float:
    """
    Calculates the critical path duration for a single phase given sampled task durations.
    This is a simplified forward pass method assuming tasks are acyclic and well-defined.
    """
    if not tasks_in_phase:
        return 0.0

    task_finish_times: Dict[str, float] = {}
    
    # Create a list of task IDs for easier lookup
    task_ids_in_phase = {task['id'] for task in tasks_in_phase}

    # Iteratively calculate finish times
    # Make a copy to modify/remove tasks as they are processed
    remaining_tasks = tasks_in_phase[:]
    
    # Safety break for potential cycles or unresolvable dependencies in malformed input
    # In a DAG, each task should be processed once. Max N iterations.
    max_iterations = len(tasks_in_phase) + 1 
    iterations = 0

    while remaining_tasks and iterations < max_iterations:
        processed_in_this_iteration_count = 0
        tasks_for_next_iteration = []

        for task in remaining_tasks:
            task_id = task['id']
            predecessors = task['predecessors']
            
            # Check if all predecessors are valid and processed
            ready_to_process = True
            max_predecessor_finish_time = 0.0

            if not predecessors: # Task has no predecessors
                pass # max_predecessor_finish_time remains 0.0
            else:
                for pred_id in predecessors:
                    if pred_id not in task_ids_in_phase:
                        # This indicates a misconfiguration: predecessor from another phase or non-existent
                        # For this simplified model, we might ignore it or treat as 0, or raise error.
                        # Assuming valid inputs: all pred_ids are in this phase or completed (implicitly 0 if from prior phase).
                        # The problem implies phases are sequential, so predecessors are within the same phase.
                        # print(f"Warning: Task {task_id} has predecessor {pred_id} not in current phase task list.")
                        # For now, let's assume valid inputs.
                        pass

                    if pred_id not in task_finish_times:
                        ready_to_process = False
                        break 
                    max_predecessor_finish_time = max(max_predecessor_finish_time, task_finish_times[pred_id])
            
            if ready_to_process:
                current_task_duration = task_durations_this_run.get(task_id, 0.0) # Get sampled duration
                task_finish_times[task_id] = max_predecessor_finish_time + current_task_duration
                processed_in_this_iteration_count += 1
            else:
                tasks_for_next_iteration.append(task) # Defer to next iteration

        remaining_tasks = tasks_for_next_iteration
        iterations += 1

        if processed_in_this_iteration_count == 0 and remaining_tasks:
            # Stalemate: No tasks were processed, but some remain.
            # This indicates a cycle or missing predecessor information.
            # print(f"Warning: Critical path calculation stalemate. Unprocessed tasks: {[t['id'] for t in remaining_tasks]}.")
            # This could lead to an underestimate if these tasks are ignored.
            # For this prompt, we assume valid acyclic task dependencies within a phase.
            break # Exit loop to prevent infinite loop

    if not task_finish_times: # No tasks processed or no tasks at all
        return 0.0
        
    return max(task_finish_times.values()) if task_finish_times else 0.0


# -------------------- Main Simulation Function --------------------

def calculate_scenarios(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs an enhanced Monte Carlo simulation for project schedule and cost.
    """
    project_name: str = input_data["projectName"]
    team_size: int = input_data["teamSize"]
    daily_cost_per_member: float = input_data["dailyCost"] # Renamed for clarity
    sim_runs: int = input_data["simRuns"]
    phases_data: List[Dict[str, Any]] = input_data["phases"]
    risks_data: List[Dict[str, Any]] = input_data["risks"]

    duration_results: List[float] = []
    cost_results: List[float] = []
    
    # Initialize risk impact tracker
    risk_impact_tracker: Dict[str, Dict[str, Any]] = {
        risk['id']: {
            'name': risk['name'],
            'occurrences': 0,
            'totalDays': 0.0,
            'totalCost': 0.0
        } for risk in risks_data
    }

    # --- Simulation Loop ---
    for _ in range(sim_runs):
        current_run_project_duration: float = 0.0
        current_run_additional_cost_from_risks: float = 0.0 # Direct cost impact from risks

        # Create a deep copy of risks for this run to modify probabilities due to correlation
        # This ensures original risk data (and probabilities) remains unchanged for next run's correlation step
        current_run_active_risks = copy.deepcopy(risks_data)

        # --- 1. Handle Risk Correlations (Modify probabilities for this run) ---
        # Determine which original risks occurred to trigger correlations
        triggered_correlation_source_risk_ids: set[str] = set()
        for original_risk_def in risks_data: # Use original probabilities for checking occurrence FOR correlation
            if random.random() < original_risk_def['probability']:
                triggered_correlation_source_risk_ids.add(original_risk_def['id'])
        
        # Apply probability increases based on triggered correlations
        for original_risk_def in risks_data: # Iterate original again to find defined correlations
            if original_risk_def['id'] in triggered_correlation_source_risk_ids:
                correlation_info = original_risk_def.get('correlates', {})
                target_risk_id = correlation_info.get('risk_id')
                prob_increase = correlation_info.get('prob_increase')

                if target_risk_id and prob_increase is not None:
                    # Find the target risk in current_run_active_risks and modify its probability
                    for r_active in current_run_active_risks:
                        if r_active['id'] == target_risk_id:
                            r_active['probability'] = min(1.0, r_active['probability'] + prob_increase)
                            break
        
        # --- 2. Calculate Phase Durations (Project Baseline Duration for this run) ---
        phase_offset = 0.0 # Not strictly needed as phases are sequential duration sum
        for phase in phases_data:
            phase_tasks = phase['tasks']
            
            # Sample task durations for all tasks in this phase for this run
            task_durations_this_run_for_phase: Dict[str, float] = {}
            for task_detail in phase_tasks:
                task_durations_this_run_for_phase[task_detail['id']] = _calculate_task_duration_sample(task_detail)
            
            phase_duration = _calculate_phase_critical_path_duration(phase_tasks, task_durations_this_run_for_phase)
            current_run_project_duration += phase_duration
            # phase_offset += phase_duration # If we needed absolute start/end times for tasks

        # --- 3. Apply Risk Impacts (to duration and cost) ---
        for active_risk in current_run_active_risks: # Use the (potentially modified) risks for this run
            if random.random() < active_risk['probability']: # Check if this risk occurs
                risk_id = active_risk['id']
                tracker_entry = risk_impact_tracker[risk_id]
                tracker_entry['occurrences'] += 1

                # Apply day impact
                sampled_impact_days = 0.0
                if active_risk.get('impactDays'):
                    sampled_impact_days = _calculate_risk_impact_sample(active_risk['impactDays'])
                    current_run_project_duration += sampled_impact_days
                    tracker_entry['totalDays'] += sampled_impact_days
                
                # Apply cost impact
                sampled_impact_cost = 0.0
                if active_risk.get('impactCost'):
                    sampled_impact_cost = _calculate_risk_impact_sample(active_risk['impactCost'])
                    current_run_additional_cost_from_risks += sampled_impact_cost
                    tracker_entry['totalCost'] += sampled_impact_cost

        # --- 4. Calculate Final Cost for the run ---
        # Cost = (Total Duration * Team Size * Daily Cost per Member) + Additional Direct Risk Costs
        total_labor_cost = current_run_project_duration * team_size * daily_cost_per_member
        final_run_cost = total_labor_cost + current_run_additional_cost_from_risks
        
        # --- 5. Store Results for this run ---
        duration_results.append(current_run_project_duration)
        cost_results.append(final_run_cost)

    # --- Analyze Results (After all simulation runs) ---
    duration_results_np = np.array(duration_results)
    cost_results_np = np.array(cost_results)

    duration_analysis = {
        "mean": np.mean(duration_results_np),
        "p5": np.percentile(duration_results_np, 5),
        "p10": np.percentile(duration_results_np, 10), 
        "p50": np.percentile(duration_results_np, 50), # Median
        "p80": np.percentile(duration_results_np, 80),
        "p90": np.percentile(duration_results_np, 90), 
        "p95": np.percentile(duration_results_np, 95)
    }
    cost_analysis = {
        "mean": np.mean(cost_results_np),
        "p5": np.percentile(cost_results_np, 5),
        "p10": np.percentile(cost_results_np, 10), 
        "p50": np.percentile(cost_results_np, 50), # Median
        "p80": np.percentile(cost_results_np, 80),
        "p90": np.percentile(cost_results_np, 90), 
        "p95": np.percentile(cost_results_np, 95)
    }

    # Top Risk Drivers
    # Convert tracker dict to list of dicts for sorting and output
    risk_drivers_list = []
    for r_id, r_data in risk_impact_tracker.items():
        risk_drivers_list.append({
            "id": r_id,
            "name": r_data["name"],
            "occurrences": r_data["occurrences"],
            "totalImpactDays": r_data["totalDays"],
            "totalImpactCost": r_data["totalCost"]
        })

    top_risk_drivers_by_days = sorted(
        risk_drivers_list, key=lambda x: x['totalImpactDays'], reverse=True
    )[:3]
    # Select only relevant fields for the "byDays" output
    top_risk_drivers_by_days_output = [
        {"id": r["id"], "name": r["name"], "occurrences": r["occurrences"], "totalImpactDays": r["totalImpactDays"]}
        for r in top_risk_drivers_by_days
    ]


    top_risk_drivers_by_cost = sorted(
        risk_drivers_list, key=lambda x: x['totalImpactCost'], reverse=True
    )[:3]
    # Select only relevant fields for the "byCost" output
    top_risk_drivers_by_cost_output = [
        {"id": r["id"], "name": r["name"], "occurrences": r["occurrences"], "totalImpactCost": r["totalImpactCost"]}
        for r in top_risk_drivers_by_cost
    ]


    # Distribution Sample
    sample_size = min(100, sim_runs)
    if sim_runs > 0: # Ensure there are results to sample from
        sample_indices = random.sample(range(sim_runs), sample_size)
        duration_sample = [duration_results[i] for i in sample_indices]
        cost_sample = [cost_results[i] for i in sample_indices]
    else:
        duration_sample = []
        cost_sample = []

    distribution_sample = {
        "durations": duration_sample,
        "costs": cost_sample
    }

    # --- Construct final output dictionary ---
    output = {
        "projectName": project_name,
        "simulationRuns": sim_runs,
        "durationAnalysis": duration_analysis,
        "costAnalysis": cost_analysis,
        "topRiskDriversByDays": top_risk_drivers_by_days_output,
        "topRiskDriversByCost": top_risk_drivers_by_cost_output,
        "distributionSample": distribution_sample
    }
    
    return output


# -------------------- Main Execution Block --------------------

if __name__ == "__main__":
    # Example with new distribution structure
    example_input_data_new = {
        "projectName": "Advanced CRM Rollout",
        "teamSize": 8,
        "dailyCost": 550, # Per member
        "simRuns": 5000,   # As per prompt requirement
        "phases": [
            {
                "phaseName": "Phase 1: Planning & Design",
                "tasks": [
                    {
                        "id": "T1", 
                        "name": "Requirements Gathering", 
                        "duration_params": {
                            "type": "PERT",
                            "optimistic": 8,
                            "most_likely": 10,
                            "pessimistic": 15
                        },
                        "predecessors": []
                    },
                    {
                        "id": "T2", 
                        "name": "Architecture Design", 
                        "duration_params": {
                            "type": "Normal",
                            "mean": 15,
                            "std_dev": 3
                        },
                        "predecessors": ["T1"]
                    },
                    {
                        "id": "T3", 
                        "name": "UI/UX Mockups", 
                        "duration_params": {
                            "type": "Triangular",
                            "optimistic": 7,
                            "most_likely": 10,
                            "pessimistic": 14
                        },
                        "predecessors": ["T1"]
                    }
                ]
            },
            {
                "phaseName": "Phase 2: Development",
                "tasks": [
                    {
                        "id": "T4", 
                        "name": "Backend Dev", 
                        "duration_params": {
                            "type": "PERT",
                            "optimistic": 30,
                            "most_likely": 40,
                            "pessimistic": 60
                        },
                        "predecessors": ["T2"]
                    },
                    {
                        "id": "T5", 
                        "name": "Frontend Dev", 
                        "duration_params": {
                            "type": "Uniform",
                            "min": 25,
                            "max": 55
                        },
                        "predecessors": ["T3"]
                    },
                    {
                        "id": "T6", 
                        "name": "Integration", 
                        "duration_params": {
                            "type": "Normal",
                            "mean": 15,
                            "std_dev": 2.5
                        },
                        "predecessors": ["T4", "T5"]
                    }
                ]
            }
        ],
        "risks": [
            {
                "id": "R1",
                "name": "Key Supplier Delay",
                "probability": 0.15,
                "impactDays": {
                    "type": "PERT",
                    "optimistic": 10,
                    "most_likely": 20,
                    "pessimistic": 30
                },
                "impactCost": {
                    "type": "Normal",
                    "mean": 10000,
                    "std_dev": 2500
                },
                "correlates": {"risk_id": "R3", "prob_increase": 0.25}
            },
            {
                "id": "R2",
                "name": "Lead Developer Leaves",
                "probability": 0.05,
                "impactDays": {
                    "type": "Triangular",
                    "optimistic": 25,
                    "most_likely": 35,
                    "pessimistic": 50
                },
                "impactCost": {
                    "type": "Uniform",
                    "min": 15000,
                    "max": 30000
                },
                "correlates": {}
            },
            {
                "id": "R3",
                "name": "Budget Overrun",
                "probability": 0.10,
                "impactDays": {
                    "type": "Normal",
                    "mean": 5,
                    "std_dev": 2
                },
                "impactCost": {
                    "type": "PERT",
                    "optimistic": 10000,
                    "most_likely": 15000,
                    "pessimistic": 25000
                },
                "correlates": {}
            }
        ]
    }

    # Example with old structure (for backward compatibility test)
    example_input_data_old = {
        "projectName": "Legacy Project",
        "teamSize": 5,
        "dailyCost": 600,
        "simRuns": 1000,
        "phases": [
            {
                "phaseName": "Phase 1",
                "tasks": [
                    {"id": "T1", "name": "Task 1", "best": 5, "likely": 7, "worst": 10, "predecessors": []},
                    {"id": "T2", "name": "Task 2", "best": 8, "likely": 10, "worst": 15, "predecessors": ["T1"]}
                ]
            }
        ],
        "risks": [
            {
                "id": "R1",
                "name": "Risk 1",
                "probability": 0.2,
                "impactDays": {"best": 5, "likely": 10, "worst": 15},
                "impactCost": {"best": 5000, "likely": 7500, "worst": 10000},
                "correlates": {}
            }
        ]
    }

    print("Running Monte Carlo Simulation with new distribution structure...")
    simulation_results = calculate_scenarios(example_input_data_new)
    print("\nSimulation Results:")
    # Pretty print the JSON output
    print(json.dumps(simulation_results, indent=4))

    # Example of accessing specific parts of the result:
    print(f"\nProject Name: {simulation_results['projectName']}")
    print(f"Simulated Mean Duration: {simulation_results['durationAnalysis']['mean']:.2f} days")
    print(f"Simulated P10 Duration: {simulation_results['durationAnalysis']['p10']:.2f} days")
    print(f"Simulated P90 Duration: {simulation_results['durationAnalysis']['p90']:.2f} days")
    print(f"Simulated P95 Duration: {simulation_results['durationAnalysis']['p95']:.2f} days")
    print(f"Simulated Mean Cost: ${simulation_results['costAnalysis']['mean']:,.2f}")
    print(f"Simulated P10 Cost: ${simulation_results['costAnalysis']['p10']:,.2f}")
    print(f"Simulated P90 Cost: ${simulation_results['costAnalysis']['p90']:,.2f}")
    print(f"Simulated P95 Cost: ${simulation_results['costAnalysis']['p95']:,.2f}")

    print("\nTop Risk Drivers by Days:")
    for risk in simulation_results['topRiskDriversByDays']:
        print(f"  - {risk['name']} (ID: {risk['id']}): Occurrences: {risk['occurrences']}, Total Impact: {risk['totalImpactDays']:.2f} days")

    print("\nTop Risk Drivers by Cost:")
    for risk in simulation_results['topRiskDriversByCost']:
        print(f"  - {risk['name']} (ID: {risk['id']}): Occurrences: {risk['occurrences']}, Total Impact: ${risk['totalImpactCost']:,.2f}")

    # Test backward compatibility
    print("\n\nTesting backward compatibility with old structure...")
    simulation_results_old = calculate_scenarios(example_input_data_old)
    print(f"Legacy Project Mean Duration: {simulation_results_old['durationAnalysis']['mean']:.2f} days")
    print("Backward compatibility test successful!")