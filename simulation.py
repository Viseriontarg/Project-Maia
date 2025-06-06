# simulation.py
import numpy as np
import random
import copy
import json # For pretty printing in __main__
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import norm, uniform, triang, beta

# -------------------- PERT Distribution Implementation --------------------
# (pert_rvs function remains the same)
def pert_rvs(a, b, c, size=None):
    """
    Generate random variates from a PERT distribution.
    Parameters: a: optimistic, b: most likely, c: pessimistic, size: output shape
    Returns: Random variates from PERT distribution
    """
    if not isinstance(a, (int, float)): a = 0
    if not isinstance(b, (int, float)): b = 0
    if not isinstance(c, (int, float)): c = 0
        
    if a > c: a, c = c, a
    if a == c: return a if size is None else np.full(size, a)
    # Clamp b to be within [a, c] if it's outside
    b = max(a, min(c, b))

    mean = (a + 4*b + c) / 6
    alpha = 1 + 4 * (b - a) / (c - a) if a != b else 1.0
    beta_param = 1 + 4 * (c - b) / (c - a) if c!=b else 1.0
    alpha = max(alpha, 0.1)
    beta_param = max(beta_param, 0.1)
    beta_samples = beta.rvs(alpha, beta_param, size=size)
    return a + (c - a) * beta_samples

# -------------------- Helper Functions --------------------
# (_sample_from_distribution, _get_triangular_sample, _calculate_task_duration_sample,
#  _calculate_risk_impact_sample, _calculate_phase_critical_path_duration remain the same)
def _safe_float(value, default=0.0):
     try:
         return float(value)
     except (ValueError, TypeError):
         return default
         
def _sample_from_distribution(dist_params: Dict[str, Any]) -> float:
    """ Samples from a specified distribution. """
    if not isinstance(dist_params, dict): return 0.0
    dist_type = dist_params.get('type', '').upper()
    if dist_type == 'PERT':
        return pert_rvs(_safe_float(dist_params.get('optimistic')),
                        _safe_float(dist_params.get('most_likely')),
                        _safe_float(dist_params.get('pessimistic')))
    elif dist_type == 'NORMAL':
        mean = _safe_float(dist_params.get('mean'))
        std_dev = _safe_float(dist_params.get('std_dev'))
        if std_dev <= 0: return mean
        return max(0, norm.rvs(loc=mean, scale=std_dev)) # Ensure non-negative
    elif dist_type == 'UNIFORM':
        min_val = _safe_float(dist_params.get('min'))
        max_val = _safe_float(dist_params.get('max'))
        if min_val >= max_val: return min_val
        return uniform.rvs(loc=min_val, scale=max_val - min_val)
    elif dist_type == 'TRIANGULAR':
        optimistic = _safe_float(dist_params.get('optimistic'))
        most_likely = _safe_float(dist_params.get('most_likely'))
        pessimistic = _safe_float(dist_params.get('pessimistic'))
        if optimistic > pessimistic: optimistic, pessimistic = pessimistic, optimistic
        most_likely = max(optimistic, min(pessimistic, most_likely)) # Clamp
        if optimistic == pessimistic: return optimistic
        loc = optimistic
        scale = pessimistic - optimistic
        c_param = (most_likely - optimistic) / scale
        c_param = max(0, min(1, c_param))
        return triang.rvs(c_param, loc=loc, scale=scale)
    # else: raise ValueError(f"Unknown or missing distribution type: {dist_type}")
    else: return 0.0 # Return 0 for unknown/missing types or bad params


def _get_triangular_sample(best: float, likely: float, worst: float) -> float:
    """ Legacy function for backward compatibility."""
    dist_params = {'type': 'TRIANGULAR','optimistic': best,'most_likely': likely,'pessimistic': worst}
    return _sample_from_distribution(dist_params)

def _calculate_task_duration_sample(task_details: Dict[str, Any]) -> float:
    """Calculates a single task duration sample."""
    if not isinstance(task_details, dict): return 0.0
    if 'duration_params' in task_details and isinstance(task_details.get('duration_params'), dict):
        return _sample_from_distribution(task_details['duration_params'])
    elif all(key in task_details for key in ['best', 'likely', 'worst']):
        return _get_triangular_sample(task_details.get('best'), task_details.get('likely'), task_details.get('worst'))
    # else: raise ValueError("Task details malformed")
    else: return 0.0

def _calculate_risk_impact_sample(impact_details: Dict[str, Any]) -> float:
    """Calculates a single risk impact sample."""
    if not isinstance(impact_details, dict): return 0.0
    if 'type' in impact_details:
        return _sample_from_distribution(impact_details)
    elif all(key in impact_details for key in ['best', 'likely', 'worst']):
        return _get_triangular_sample(impact_details.get('best'), impact_details.get('likely'), impact_details.get('worst'))
    # If no distribution is specified (e.g., empty dict), return 0 impact
    # else: raise ValueError("Impact details malformed")
    else: return 0.0


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
    # task_ids_in_phase = {task.get('id') for task in tasks_in_phase if task.get('id')} # IDs might be missing
    remaining_tasks = [t for t in tasks_in_phase if isinstance(t, dict) and 'id' in t] # Filter valid
    if not remaining_tasks: return 0.0
        
    max_iterations = len(remaining_tasks) * 2 # Safety break
    iterations = 0

    while remaining_tasks and iterations < max_iterations:
        processed_in_this_iteration_count = 0
        tasks_for_next_iteration = []

        for task in remaining_tasks:
            task_id = task['id']
            predecessors = task.get('predecessors', [])
            if not isinstance(predecessors, list): predecessors = [] # Handle malformed data
                
            ready_to_process = True
            max_predecessor_finish_time = 0.0

            if predecessors:
                for pred_id in predecessors:
                     # Check if the predecessor is *both* a string/valid ID AND it has been processed
                    if not isinstance(pred_id, str) or pred_id not in task_finish_times:
                         # if a predecessor exists but hasn't finished, we can't process
                         # if a predecessor is listed but not in *this phase's tasks at all*, or isn't a string, ignore it.
                         # check if pred_id is actually a task we expect to process
                         is_task_in_phase = any(t.get('id') == pred_id for t in tasks_in_phase)
                         if is_task_in_phase and pred_id not in task_finish_times:
                              ready_to_process = False
                              break
                    else: # Predecessor exists and is finished
                       max_predecessor_finish_time = max(max_predecessor_finish_time, task_finish_times.get(pred_id, 0.0))

            if ready_to_process:
                current_task_duration = task_durations_this_run.get(task_id, 0.0)
                task_finish_times[task_id] = max_predecessor_finish_time + max(0, current_task_duration) # No negative duration
                processed_in_this_iteration_count += 1
            else:
                tasks_for_next_iteration.append(task)

        remaining_tasks = tasks_for_next_iteration
        iterations += 1
        if processed_in_this_iteration_count == 0 and remaining_tasks: break # Stalemate
            
    # Add tasks that were never processed (e.g. no predecessors, no successors, bad ID) 
    # to ensure their duration is at least accounted for if they are isolated
    final_max_time = max(task_finish_times.values()) if task_finish_times else 0.0
    # for task in tasks_in_phase:
    #      if task.get('id') and task.get('id') not in task_finish_times:
    #           final_max_time = max(final_max_time, task_durations_this_run.get(task.get('id'), 0.0))

    return final_max_time


# -------------------- Main Simulation Function --------------------

def calculate_scenarios(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs an enhanced Monte Carlo simulation for project schedule and cost.
    """
    if not isinstance(input_data, dict): input_data = {} # handle None input
        
    project_name: str = input_data.get("projectName", "Unnamed Project")
    team_size_raw = input_data.get("teamSize", 1)
    team_size: int = int(team_size_raw) if isinstance(team_size_raw, (int, float)) and team_size_raw is not None else 1
    
    daily_cost_raw = input_data.get("dailyCost", 0)
    daily_cost_per_member: float = float(daily_cost_raw) if isinstance(daily_cost_raw, (int, float)) and daily_cost_raw is not None else 0.0

    sim_runs_raw = input_data.get("simRuns", 1000)
    # Ensure sim_runs is a positive integer, default 1000 if invalid
    sim_runs = int(sim_runs_raw) if isinstance(sim_runs_raw, (int, float)) and sim_runs_raw is not None and sim_runs_raw > 0 else 1000
    if sim_runs <=0: sim_runs = 1 # Must run at least once

    phases_data: List[Dict[str, Any]] = input_data.get("phases", [])
    if not isinstance(phases_data, list): phases_data = []
        
    risks_data: List[Dict[str, Any]] = input_data.get("risks", [])
    if not isinstance(risks_data, list): risks_data = []
    # Filter out any non-dict elements and ensure 'id' and 'name' exist
    risks_data = [
        r for r in risks_data 
        if isinstance(r, dict) and r.get('id') is not None and r.get('name') is not None
     ]

    duration_results: List[float] = []
    cost_results: List[float] = []
    
    risk_impact_tracker: Dict[str, Dict[str, Any]] = {
        risk['id']: {
            'name': risk.get('name', f"Risk {risk.get('id','?')}"), # Use name or fallback
            'occurrences': 0, 
            'totalDays': 0.0, # <-- Tracker key
            'totalCost': 0.0  # <-- Tracker key
        } for risk in risks_data
    }

    # --- Simulation Loop ---
    for i in range(sim_runs):
       # if i % 1000 == 0 and i > 0: print(f"   ... simulation run {i}...") # progress indicator
        current_run_project_duration: float = 0.0
        current_run_additional_cost_from_risks: float = 0.0
        
        # Make a deepcopy only if there are correlations defined to save memory/perf
        has_correlations = any('correlates' in r and r['correlates'] for r in risks_data)
        current_run_risks_data = copy.deepcopy(risks_data) if has_correlations else risks_data


        # --- 1. Handle Risk Correlations ---
        if has_correlations:
          triggered_correlation_source_risk_ids: set[str] = set()
          # Check occurrences based on *original* probabilities first
          for original_risk_def in risks_data:
              prob = _safe_float(original_risk_def.get('probability', 0))
              if prob > 0 and random.random() < prob:
                  triggered_correlation_source_risk_ids.add(original_risk_def['id'])
          
          # Apply increases to the deep-copied list for this run
          correlated_risk_map = {r['id']: r for r in current_run_risks_data}
          for original_risk_def in risks_data: # iterate original to get correlation defs
             if original_risk_def['id'] in triggered_correlation_source_risk_ids:
                  correlation_info = original_risk_def.get('correlates', {})
                  target_risk_id = correlation_info.get('risk_id')
                  prob_increase = _safe_float(correlation_info.get('prob_increase'))

                  if target_risk_id and prob_increase > 0 and target_risk_id in correlated_risk_map:
                       target_risk = correlated_risk_map[target_risk_id]
                       current_prob = _safe_float(target_risk.get('probability', 0))
                       target_risk['probability'] = min(1.0, current_prob + prob_increase)
        
        # --- 2. Calculate Phase Durations ---
        for phase in phases_data:
            if not isinstance(phase, dict): continue
            phase_tasks = phase.get('tasks', [])
            if not isinstance(phase_tasks, list): continue
                
            task_durations_this_run_for_phase: Dict[str, float] = {}
            for task_detail in phase_tasks:
                 if isinstance(task_detail, dict) and task_detail.get('id'):
                    task_durations_this_run_for_phase[task_detail['id']] = _calculate_task_duration_sample(task_detail)

            phase_duration = _calculate_phase_critical_path_duration(phase_tasks, task_durations_this_run_for_phase)
            current_run_project_duration += phase_duration

        # --- 3. Apply Risk Impacts ---
        # Use current_run_risks_data which might have modified probabilities
        for active_risk in current_run_risks_data: 
             prob = _safe_float(active_risk.get('probability', 0))
             if prob > 0 and random.random() < prob:
                risk_id = active_risk['id']
                # Ensure risk_id is in tracker (could be missing if risks_data was weird)
                if risk_id not in risk_impact_tracker: continue
                    
                tracker_entry = risk_impact_tracker[risk_id]
                tracker_entry['occurrences'] += 1

                if active_risk.get('impactDays'):
                    sampled_impact_days = _calculate_risk_impact_sample(active_risk['impactDays'])
                    current_run_project_duration += sampled_impact_days
                    tracker_entry['totalDays'] += sampled_impact_days # Use tracker key
                
                if active_risk.get('impactCost'):
                    sampled_impact_cost = _calculate_risk_impact_sample(active_risk['impactCost'])
                    current_run_additional_cost_from_risks += sampled_impact_cost
                    tracker_entry['totalCost'] += sampled_impact_cost # Use tracker key

        # --- 4. Calculate Final Cost for the run ---
        current_run_project_duration = max(0, current_run_project_duration) # no negative duration
        total_labor_cost = current_run_project_duration * team_size * daily_cost_per_member
        final_run_cost = total_labor_cost + current_run_additional_cost_from_risks
        
        # --- 5. Store Results for this run ---
        duration_results.append(current_run_project_duration)
        cost_results.append(final_run_cost)

    # --- Analyze Results ---
    duration_results_np = np.array(duration_results) if duration_results else np.array([0])
    cost_results_np = np.array(cost_results) if cost_results else np.array([0])
    
    def get_percentile(data, p):
       try:
           # Handle potential NaN or errors if data is empty or all same value
           if not data.any() or len(data) < 2: return 0.0
           val = np.percentile(data, p)
           return val if np.isfinite(val) else 0.0
       except:
            return 0.0
            
    mean_dur = np.mean(duration_results_np) if duration_results_np.any() else 0.0
    mean_cost = np.mean(cost_results_np) if cost_results_np.any() else 0.0

    duration_analysis = {
        "mean": mean_dur if np.isfinite(mean_dur) else 0.0,
        "p5": get_percentile(duration_results_np, 5),
        "p10": get_percentile(duration_results_np, 10),
        "p50": get_percentile(duration_results_np, 50),
        "p80": get_percentile(duration_results_np, 80),
        "p90": get_percentile(duration_results_np, 90),
        "p95": get_percentile(duration_results_np, 95),
    }
    cost_analysis = {
         "mean": mean_cost if np.isfinite(mean_cost) else 0.0,
         "p5": get_percentile(cost_results_np, 5),
         "p10": get_percentile(cost_results_np, 10),
         "p50": get_percentile(cost_results_np, 50),
         "p80": get_percentile(cost_results_np, 80),
         "p90": get_percentile(cost_results_np, 90),
         "p95": get_percentile(cost_results_np, 95),
    }
    
    # --- !!! FIX STARTS HERE: Restore original map/sort/shape logic !!! ---
    
    # 1. MAP: Convert tracker dict (with totalDays/Cost) to list for sorting (with totalImpactDays/Cost)
    risk_drivers_list = []
    for r_id, r_data in risk_impact_tracker.items():
         risk_drivers_list.append({
             "id": r_id,
             "name": r_data["name"],
             "occurrences": r_data["occurrences"],
             "totalImpactDays": r_data["totalDays"], # MAP: totalDays -> totalImpactDays
             "totalImpactCost": r_data["totalCost"]  # MAP: totalCost -> totalImpactCost
         })

    # 2. SORT: Sort the mapped list
    top_risk_drivers_by_days_sorted = sorted(
        risk_drivers_list, 
        key=lambda x: x.get('totalImpactDays', 0.0), # SORT using mapped key
        reverse=True
    )[:3]
    
    top_risk_drivers_by_cost_sorted = sorted(
        risk_drivers_list, 
        key=lambda x: x.get('totalImpactCost', 0.0), # SORT using mapped key
        reverse=True
    )[:3]

    # 3. SHAPE/FILTER: Create final output lists with only relevant keys
    top_risk_drivers_by_days_output = [
        {"id": r["id"], "name": r["name"], "occurrences": r["occurrences"], "totalImpactDays": r.get("totalImpactDays",0.0)}
        for r in top_risk_drivers_by_days_sorted
    ]
    top_risk_drivers_by_cost_output = [
       {"id": r["id"], "name": r["name"], "occurrences": r["occurrences"], "totalImpactCost": r.get("totalImpactCost", 0.0)}
        for r in top_risk_drivers_by_cost_sorted
     ]
      # --- !!! FIX ENDS HERE !!! ---


    sample_size = min(100, sim_runs)
    distribution_sample = {"durations": [], "costs": []}
    if sim_runs > 0 and duration_results: # Check results exist
        # Ensure sample size doesn't exceed number of available runs
        actual_sample_size = min(sample_size, len(duration_results))
        if actual_sample_size > 0:
           sample_indices = random.sample(range(len(duration_results)), actual_sample_size)
           distribution_sample["durations"] = [duration_results[i] for i in sample_indices]
           distribution_sample["costs"] = [cost_results[i] for i in sample_indices]

    # RETURN the shaped/filtered output lists
    return {
        "projectName": project_name, "simulationRuns": sim_runs,
        "durationAnalysis": duration_analysis, "costAnalysis": cost_analysis,
        # Use the final _output lists
        "topRiskDriversByDays": top_risk_drivers_by_days_output, 
        "topRiskDriversByCost": top_risk_drivers_by_cost_output,
        "distributionSample": distribution_sample
    }


# -------------------- Main Execution Block --------------------
# (remains the same)
if __name__ == "__main__":
    # Example with new distribution structure
    example_input_data_new = {
        "projectName": "Advanced CRM Rollout",
        "teamSize": 8, "dailyCost": 550, "simRuns": 500, # Reduced runs for quick test
        "targetBaseCost": 200000, # Added for reporting context
        "targetBaseDuration": 80, # Added for reporting context
        "phases": [{
            "phaseName": "Phase 1: Planning & Design",
            "tasks": [
                {"id": "T1", "name": "Requirements", "duration_params": {"type": "PERT", "optimistic": 8, "most_likely": 10, "pessimistic": 15}, "predecessors": []},
                {"id": "T2", "name": "Architecture", "duration_params": {"type": "Normal", "mean": 15, "std_dev": 3}, "predecessors": ["T1"]},
                {"id": "T3", "name": "UI/UX", "duration_params": {"type": "Triangular", "optimistic": 7, "most_likely": 10, "pessimistic": 14}, "predecessors": ["T1"]}
            ]}, {
            "phaseName": "Phase 2: Development",
            "tasks": [
                {"id": "T4", "name": "Backend", "duration_params": {"type": "PERT", "optimistic": 30, "most_likely": 40, "pessimistic": 60}, "predecessors": ["T2"]},
                {"id": "T5", "name": "Frontend", "duration_params": {"type": "Uniform", "min": 25, "max": 55}, "predecessors": ["T3"]},
                 # Parallel task to T5
                {"id": "T5b", "name": "API Dev", "duration_params": {"type": "PERT", "optimistic": 20, "most_likely": 30, "pessimistic": 45}, "predecessors": ["T2"]},
                 {"id": "T6", "name": "Integration", "duration_params": {"type": "Normal", "mean": 15, "std_dev": 2.5}, "predecessors": ["T4", "T5", "T5b"]} # Wait for all 3
            ]},
             {
            "phaseName": "Phase 3: Testing",
            "tasks": [
                # Task with non-existent predecessor
                 {"id": "T7", "name": "UAT", "duration_params": {"type": "PERT", "optimistic": 10, "most_likely": 15, "pessimistic": 20}, "predecessors": ["T6", "TX"]},
             ]}
            ],
        "risks": [{
            "id": "R1", "name": "Key Supplier Delay", "probability": 0.15,
            "impactDays": {"type": "PERT", "optimistic": 10, "most_likely": 20, "pessimistic": 30},
            "impactCost": {"type": "Normal", "mean": 10000, "std_dev": 2500},
            "correlates": {"risk_id": "R3", "prob_increase": 0.25}
            }, {
            "id": "R2", "name": "Lead Developer Leaves", "probability": 0.05,
            "impactDays": {"type": "Triangular", "optimistic": 25, "most_likely": 35, "pessimistic": 50},
            "impactCost": {"type": "Uniform", "min": 15000, "max": 30000},
            "correlates": {}
            }, {
            "id": "R3", "name": "Budget Overrun", "probability": 0.10,
             # Risk with no day impact
            "impactDays": {}, #{"type": "Normal", "mean": 5, "std_dev": 2},
            "impactCost": {"type": "PERT", "optimistic": 10000, "most_likely": 15000, "pessimistic": 25000},
            "correlates": {}
            },
            # Risk with no impacts at all
             {
            "id": "R4", "name": "Minor Bug", "probability": 0.80,
            "impactDays": {}, 
            "impactCost": {},
            "correlates": {}
            }
            
            ]
    }

    print("\nRunning Monte Carlo Simulation with new distribution structure...")
    try:
      simulation_results = calculate_scenarios(example_input_data_new)
      print("\nSimulation Results (New):")
      # Use a custom encoder for numpy types
      class NumpyEncoder(json.JSONEncoder):
          def default(self, obj):
              if isinstance(obj, (np.integer, np.floating)): return obj.item()
              if isinstance(obj, np.ndarray): return obj.tolist()
              return super(NumpyEncoder, self).default(obj)
      print(json.dumps(simulation_results, indent=4, cls=NumpyEncoder))
       # Example check
      print("\nRisk Check:")
      for r in simulation_results["topRiskDriversByDays"]:
           print(f"  Day Driver: {r.get('name')}, Keys: {list(r.keys())}")
      for r in simulation_results["topRiskDriversByCost"]:
            print(f"  Cost Driver: {r.get('name')}, Keys: {list(r.keys())}")

    except Exception as e:
         print(f"\n!!! Error during new simulation: {e}")
         import traceback
         traceback.print_exc()


    # Example with old structure (for backward compatibility test)
    example_input_data_old = {
        "projectName": "Legacy Project", "teamSize": 5, "dailyCost": 600, "simRuns": 100,
        "phases": [{"phaseName": "Phase 1", "tasks": [
            {"id": "T1", "name": "Task 1", "best": 5, "likely": 7, "worst": 10, "predecessors": []},
            {"id": "T2", "name": "Task 2", "best": 8, "likely": 10, "worst": 15, "predecessors": ["T1"]}]}],
        "risks": [{"id": "R1", "name": "Risk 1", "probability": 0.2, "correlates": {},
            "impactDays": {"best": 5, "likely": 10, "worst": 15},
            "impactCost": {"best": 5000, "likely": 7500, "worst": 10000}}]
    }
    
    print("\n\nTesting backward compatibility with old structure...")
    try:
       simulation_results_old = calculate_scenarios(example_input_data_old)
       print(f"Legacy Project Mean Duration: {simulation_results_old['durationAnalysis']['mean']:.2f} days")
       print(f"Legacy Project Top Risk Days Keys: {list(simulation_results_old['topRiskDriversByDays'][0].keys()) if simulation_results_old['topRiskDriversByDays'] else 'N/A'}")
       print("Backward compatibility test successful!")
    except Exception as e:
        print(f"\n!!! Error during legacy simulation: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nTesting simulation with empty input...")
    try:
       simulation_results_empty = calculate_scenarios({})
       print(f"Empty Project P50 Duration: {simulation_results_empty['durationAnalysis']['p50']:.2f} days")
       print("Empty input test successful!")
    except Exception as e:
        print(f"\n!!! Error during empty simulation: {e}")
        import traceback
        traceback.print_exc()