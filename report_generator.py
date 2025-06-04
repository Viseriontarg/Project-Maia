

import pandas as pd
import numpy as np
from openpyxl.styles import Font, Border, Side, Alignment, NamedStyle
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference, Series
from openpyxl.chart.shapes import GraphicalProperties

# --- Predefined Named Styles ---
currency_style = NamedStyle(name='currency', number_format='$#,##0.00')
integer_style = NamedStyle(name='integer', number_format='#,##0')
days_style = NamedStyle(name='days', number_format='#,##0" days"') # Custom format for days
percent_style = NamedStyle(name='percent_decimal', number_format='0.00%') # For values like 0.15 -> 15.00%

def _add_named_styles(workbook):
    """Adds predefined named styles to the workbook if they don't exist."""
    styles_to_add = [currency_style, integer_style, days_style, percent_style]
    for style in styles_to_add:
        if style.name not in workbook.style_names:
            workbook.add_named_style(style)

# --- Helper function for NumPy data type conversion ---
def convert_numpy_types(data):
    """
    Recursively converts NumPy data types (float64, int64, ndarray)
    in a dictionary or list to standard Python types.
    """
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return convert_numpy_types(data.tolist())
    return data

# --- Helper function for styling DataFrames written to Excel ---
def _style_dataframe_output(ws, df_num_rows, df_num_cols, start_row=1, start_col=1,
                            col_styles=None, auto_width_padding=3, header_alignment='left',
                            workbook=None):
    header_font = Font(bold=True, name='Calibri', size=11)
    thin_border_side = Side(border_style="thin", color="000000")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)

    for c_offset in range(df_num_cols):
        cell = ws.cell(row=start_row, column=start_col + c_offset)
        cell.font = header_font
        cell.border = cell_border
        cell.alignment = Alignment(horizontal=header_alignment, vertical='center')

    for r_offset in range(df_num_rows):
        row_current = start_row + 1 + r_offset
        for c_offset in range(df_num_cols):
            col_current = start_col + c_offset
            cell = ws.cell(row=row_current, column=col_current)
            cell.border = cell_border
            if col_styles and c_offset in col_styles:
                style_spec = col_styles[c_offset]
                if isinstance(style_spec, str) and workbook and style_spec in workbook.style_names:
                     cell.style = style_spec
                elif isinstance(style_spec, str):
                    cell.number_format = style_spec

    if workbook:
        for c_offset in range(df_num_cols):
            column_idx_1_based = start_col + c_offset
            column_letter = get_column_letter(column_idx_1_based)
            max_len = 0
            header_cell_value = ws.cell(row=start_row, column=column_idx_1_based).value
            if header_cell_value:
                max_len = len(str(header_cell_value))
            for r_offset in range(df_num_rows):
                cell = ws.cell(row=start_row + 1 + r_offset, column=column_idx_1_based)
                if cell.value is not None:
                    val_str = str(cell.value)
                    effective_number_format = cell.number_format
                    if cell.style != 'Normal' and cell.style in workbook._named_styles:
                        named_style_object = workbook._named_styles[cell.style]
                        effective_number_format = named_style_object.number_format
                    if effective_number_format and isinstance(cell.value, (int, float)):
                        if '$' in effective_number_format and currency_style.number_format in effective_number_format:
                             val_str = f"${cell.value:,.2f}"
                        elif '%' in effective_number_format and effective_number_format.endswith('%'):
                            val_str = f"{cell.value*100:.2f}%"
                        elif '" days"' in effective_number_format:
                             val_str = f"{cell.value:,.0f} days"
                        elif effective_number_format == integer_style.number_format or effective_number_format == '#,##0':
                            val_str = f"{cell.value:,.0f}"
                    current_len = len(val_str)
                    if current_len > max_len:
                        max_len = current_len
            ws.column_dimensions[column_letter].width = max_len + auto_width_padding


# --- Sheet Creation Functions ---
def _create_summary_sheet(writer, input_data, simulation_results):
    """Creates the 'Summary' sheet."""
    ws_name = "Summary"
    ws = writer.sheets[ws_name]
    wb = writer.book

    summary_items = []
    # Project Overview (Data from input_data)
    summary_items.append(("Project Name", input_data.get("projectName", "N/A")))
    
    # User's original targets (NEW)
    user_target_cost = input_data.get('targetBaseCost')
    user_target_duration = input_data.get('targetBaseDuration')
    summary_items.append(("User Target Base Cost", user_target_cost if user_target_cost is not None else "N/A"))
    summary_items.append(("User Target Base Duration", user_target_duration if user_target_duration is not None else "N/A"))
    
    # AI-generated parameters
    summary_items.append(("Team Size", input_data.get("teamSize", "N/A")))
    summary_items.append(("Daily Cost", input_data.get("dailyCost", "N/A")))

    # Key Results (Data from simulation_results)
    duration_analysis = simulation_results.get("durationAnalysis", {})
    cost_analysis = simulation_results.get("costAnalysis", {})

    # --- ADD P10 ---
    summary_items.append(("P10 (Optimistic) Duration", duration_analysis.get("p10", "N/A")))
    summary_items.append(("P10 (Optimistic) Cost", cost_analysis.get("p10", "N/A")))

    # --- Existing P50 ---
    summary_items.append(("P50 (Likely) Duration", duration_analysis.get("p50", "N/A")))
    summary_items.append(("P50 (Likely) Cost", cost_analysis.get("p50", "N/A")))

    # --- ADD P90 ---
    summary_items.append(("P90 (Pessimistic) Duration", duration_analysis.get("p90", "N/A")))
    summary_items.append(("P90 (Pessimistic) Cost", cost_analysis.get("p90", "N/A")))

    # --- Existing P95 ---
    summary_items.append(("P95 (Worst Case) Duration", duration_analysis.get("p95", "N/A")))
    summary_items.append(("P95 (Worst Case) Cost", cost_analysis.get("p95", "N/A")))


    # Top Risk Drivers (Data from simulation_results - Updated labels)
    top_risks_by_days_data = simulation_results.get('topRiskDriversByDays', [])
    for i in range(3):
        metric_name_days = f"Top Risk {i+1} - Total Aggregated Days Impact (All Runs)"
        if i < len(top_risks_by_days_data):
            risk = top_risks_by_days_data[i]
            name = risk.get('name', 'Unknown Risk')
            impact = risk.get('totalImpactDays', 0.0)
            summary_items.append((metric_name_days, f"{name} ({impact:.1f} days)"))
        else:
            summary_items.append((metric_name_days, "N/A"))

    top_risks_by_cost_data = simulation_results.get('topRiskDriversByCost', [])
    for i in range(3):
        metric_name_cost = f"Top Risk {i+1} - Total Aggregated Cost Impact (All Runs)"
        if i < len(top_risks_by_cost_data):
            risk = top_risks_by_cost_data[i]
            name = risk.get('name', 'Unknown Risk')
            impact = risk.get('totalImpactCost', 0.0)
            summary_items.append((metric_name_cost, f"{name} (${impact:,.0f})"))
        else:
            summary_items.append((metric_name_cost, "N/A"))

    df_summary = pd.DataFrame(summary_items, columns=["Metric", "Value"])
    df_summary.to_excel(writer, sheet_name=ws_name, index=False, startrow=0)

    # Styling for Summary table
    header_font = Font(bold=True, name='Calibri', size=11)
    thin_border_side = Side(border_style="thin", color="000000")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    for col_idx in range(1, 3):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font; cell.border = cell_border
        cell.alignment = Alignment(horizontal='left', vertical='center')
    for row_idx_df in range(len(df_summary)):
        excel_row = row_idx_df + 2
        cell_a = ws.cell(row=excel_row, column=1); cell_a.font = header_font; cell_a.border = cell_border
        cell_b = ws.cell(row=excel_row, column=2); cell_b.border = cell_border
        metric_name = df_summary.iloc[row_idx_df, 0]
        value_original = df_summary.iloc[row_idx_df, 1]
        if isinstance(value_original, (int, float)):
            # UPDATED: Include P10 and P90 in styling conditions
            p_value_cost_keywords = ["P10", "P50", "P90", "P95"]
            p_value_duration_keywords = ["P10", "P50", "P90", "P95"]

            if "Cost" in metric_name and ("Daily Cost" in metric_name or any(p_val in metric_name for p_val in p_value_cost_keywords)):
                cell_b.style = currency_style.name
            elif "User Target Base Cost" == metric_name:  # NEW styling condition
                cell_b.style = currency_style.name
            elif "Duration" in metric_name and any(p_val in metric_name for p_val in p_value_duration_keywords):
                cell_b.style = days_style.name
            elif "User Target Base Duration" == metric_name:  # NEW styling condition
                cell_b.style = days_style.name
            elif "Team Size" == metric_name:
                cell_b.style = integer_style.name
    
    # Adjusted column width to accommodate longer labels
    ws.column_dimensions[get_column_letter(1)].width = 55
    ws.column_dimensions[get_column_letter(2)].width = 30

    # Histogram Chart (Unchanged logic)
    distribution_sample = simulation_results.get('distributionSample', {})
    durations_sample = distribution_sample.get('durations', [])
    if durations_sample and len(durations_sample) > 1:
        num_bins = 10
        counts, bin_edges = np.histogram(durations_sample, bins=num_bins)
        chart_data_start_row = len(df_summary) + 4
        chart_data_start_col = 20 # Column T
        bin_labels = []
        for i in range(len(bin_edges) - 1):
            label = f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}"
            bin_labels.append(label)
            ws.cell(row=chart_data_start_row + i, column=chart_data_start_col + 1).value = label
        for i, count in enumerate(counts):
            ws.cell(row=chart_data_start_row + i, column=chart_data_start_col).value = count
        chart = BarChart(); chart.type = "col"; chart.style = 10; chart.grouping = "standard"
        chart.title = "Duration Distribution"; chart.y_axis.title = "Frequency"; chart.x_axis.title = "Duration (Days)"
        data_ref = Reference(ws, min_col=chart_data_start_col, min_row=chart_data_start_row, max_row=chart_data_start_row + len(counts) - 1)
        cat_ref = Reference(ws, min_col=chart_data_start_col + 1, min_row=chart_data_start_row, max_row=chart_data_start_row + len(bin_labels) - 1)
        series = Series(data_ref, title_from_data=False, title="Frequency")
        series.graphicalProperties = GraphicalProperties(solidFill="5B9BD5")
        chart.series.append(series); chart.set_categories(cat_ref); chart.legend = None; chart.y_axis.majorGridlines = None
        chart_anchor_cell = get_column_letter(df_summary.shape[1] + 2) + "2"
        ws.add_chart(chart, chart_anchor_cell); chart.width = 15; chart.height = 7.5
    else:
        chart_placeholder_row = len(df_summary) + 4
        ws.cell(row=chart_placeholder_row, column=1).value = "Duration distribution chart not generated (insufficient or no sample data)."

def _create_inputs_assumptions_sheet(writer, input_data):
    """Creates the 'Inputs & Assumptions' sheet using data from input_data."""
    ws_name = "Inputs & Assumptions"
    wb = writer.book
    if ws_name not in wb.sheetnames: ws = wb.create_sheet(title=ws_name)
    else: ws = wb[ws_name]
    writer.sheets[ws_name] = ws
    current_row_excel = 1
    title_font = Font(bold=True, size=14, name='Calibri')

    # --- Tasks Table ---
    tasks_for_df = []
    phases_data = input_data.get('phases', [])
    if phases_data:
        for phase in phases_data:
            phase_name = phase.get('phaseName', 'N/A')
            for task in phase.get('tasks', []):
                task_row = {
                    'Phase': phase_name,
                    'Task Name': task.get('name', 'N/A'),
                    'Distribution Type': 'N/A',
                    'Optimistic (days)': 'N/A',
                    'Most Likely (days)': 'N/A',
                    'Pessimistic (days)': 'N/A',
                    'Mean (days)': 'N/A',
                    'Std Dev (days)': 'N/A',
                    'Min (days)': 'N/A',
                    'Max (days)': 'N/A'
                }
                
                # Check for new structure with duration_params
                if 'duration_params' in task:
                    duration_params = task['duration_params']
                    dist_type = duration_params.get('type', 'N/A')
                    task_row['Distribution Type'] = dist_type
                    
                    # Populate parameters based on distribution type
                    if dist_type in ['PERT', 'Triangular']:
                        task_row['Optimistic (days)'] = duration_params.get('optimistic', 'N/A')
                        task_row['Most Likely (days)'] = duration_params.get('most_likely', 'N/A')
                        task_row['Pessimistic (days)'] = duration_params.get('pessimistic', 'N/A')
                    elif dist_type == 'Normal':
                        task_row['Mean (days)'] = duration_params.get('mean', 'N/A')
                        task_row['Std Dev (days)'] = duration_params.get('std_dev', 'N/A')
                    elif dist_type == 'Uniform':
                        task_row['Min (days)'] = duration_params.get('min', 'N/A')
                        task_row['Max (days)'] = duration_params.get('max', 'N/A')
                # Fallback to old structure
                elif all(key in task for key in ['best', 'likely', 'worst']):
                    task_row['Distribution Type'] = 'Triangular (Legacy)'
                    task_row['Optimistic (days)'] = task.get('best', 'N/A')
                    task_row['Most Likely (days)'] = task.get('likely', 'N/A')
                    task_row['Pessimistic (days)'] = task.get('worst', 'N/A')
                
                tasks_for_df.append(task_row)

    if tasks_for_df:
        df_tasks_display = pd.DataFrame(tasks_for_df)
        ws.cell(row=current_row_excel, column=1).value = "Project Tasks (from Input Data)"
        ws.cell(row=current_row_excel, column=1).font = title_font
        current_row_excel += 1
        df_tasks_display.to_excel(writer, sheet_name=ws_name, startrow=current_row_excel -1, index=False)
        
        task_col_styles = {}
        # Apply integer style to all columns containing '(days)'
        for idx, col_name in enumerate(df_tasks_display.columns):
            if '(days)' in col_name: 
                task_col_styles[idx] = integer_style.name
        
        _style_dataframe_output(ws, len(df_tasks_display), len(df_tasks_display.columns),
                                start_row=current_row_excel, start_col=1,
                                col_styles=task_col_styles, workbook=wb)
        current_row_excel += len(df_tasks_display) + 3
    else:
        ws.cell(row=current_row_excel, column=1).value = "No task data found in input_data['phases']."
        ws.cell(row=current_row_excel, column=1).font = title_font
        current_row_excel += 2

    # --- Risks Table (from Input Data) ---
    risks_for_df = []
    risks_data_input = input_data.get('risks', [])
    if risks_data_input:
        for risk in risks_data_input:
            risk_row = {
                'Risk Name': risk.get('name', 'N/A'),
                'Probability': risk.get('probability', 'N/A'),
                'Sched. Impact Dist. Type': 'N/A',
                'Optimistic Sched. Impact (days)': 'N/A',
                'Most Likely Sched. Impact (days)': 'N/A',
                'Pessimistic Sched. Impact (days)': 'N/A',
                'Mean Sched. Impact (days)': 'N/A',
                'Std Dev Sched. Impact (days)': 'N/A',
                'Min Sched. Impact (days)': 'N/A',
                'Max Sched. Impact (days)': 'N/A',
                'Cost Impact Dist. Type': 'N/A',
                'Optimistic Cost Impact ($)': 'N/A',
                'Most Likely Cost Impact ($)': 'N/A',
                'Pessimistic Cost Impact ($)': 'N/A',
                'Mean Cost Impact ($)': 'N/A',
                'Std Dev Cost Impact ($)': 'N/A',
                'Min Cost Impact ($)': 'N/A',
                'Max Cost Impact ($)': 'N/A'
            }
            
            # Process schedule impact (impactDays)
            impact_days = risk.get('impactDays', {})
            if 'type' in impact_days:
                dist_type = impact_days.get('type', 'N/A')
                risk_row['Sched. Impact Dist. Type'] = dist_type
                
                if dist_type in ['PERT', 'Triangular']:
                    risk_row['Optimistic Sched. Impact (days)'] = impact_days.get('optimistic', 'N/A')
                    risk_row['Most Likely Sched. Impact (days)'] = impact_days.get('most_likely', 'N/A')
                    risk_row['Pessimistic Sched. Impact (days)'] = impact_days.get('pessimistic', 'N/A')
                elif dist_type == 'Normal':
                    risk_row['Mean Sched. Impact (days)'] = impact_days.get('mean', 'N/A')
                    risk_row['Std Dev Sched. Impact (days)'] = impact_days.get('std_dev', 'N/A')
                elif dist_type == 'Uniform':
                    risk_row['Min Sched. Impact (days)'] = impact_days.get('min', 'N/A')
                    risk_row['Max Sched. Impact (days)'] = impact_days.get('max', 'N/A')
            # Fallback to old structure
            elif all(key in impact_days for key in ['best', 'worst']):
                risk_row['Sched. Impact Dist. Type'] = 'Triangular (Legacy)'
                risk_row['Optimistic Sched. Impact (days)'] = impact_days.get('best', 'N/A')
                risk_row['Most Likely Sched. Impact (days)'] = impact_days.get('likely', 'N/A')
                risk_row['Pessimistic Sched. Impact (days)'] = impact_days.get('worst', 'N/A')
            
            # Process cost impact (impactCost)
            impact_cost = risk.get('impactCost', {})
            if 'type' in impact_cost:
                dist_type = impact_cost.get('type', 'N/A')
                risk_row['Cost Impact Dist. Type'] = dist_type
                
                if dist_type in ['PERT', 'Triangular']:
                    risk_row['Optimistic Cost Impact ($)'] = impact_cost.get('optimistic', 'N/A')
                    risk_row['Most Likely Cost Impact ($)'] = impact_cost.get('most_likely', 'N/A')
                    risk_row['Pessimistic Cost Impact ($)'] = impact_cost.get('pessimistic', 'N/A')
                elif dist_type == 'Normal':
                    risk_row['Mean Cost Impact ($)'] = impact_cost.get('mean', 'N/A')
                    risk_row['Std Dev Cost Impact ($)'] = impact_cost.get('std_dev', 'N/A')
                elif dist_type == 'Uniform':
                    risk_row['Min Cost Impact ($)'] = impact_cost.get('min', 'N/A')
                    risk_row['Max Cost Impact ($)'] = impact_cost.get('max', 'N/A')
            # Fallback to old structure
            elif all(key in impact_cost for key in ['best', 'worst']):
                risk_row['Cost Impact Dist. Type'] = 'Triangular (Legacy)'
                risk_row['Optimistic Cost Impact ($)'] = impact_cost.get('best', 'N/A')
                risk_row['Most Likely Cost Impact ($)'] = impact_cost.get('likely', 'N/A')
                risk_row['Pessimistic Cost Impact ($)'] = impact_cost.get('worst', 'N/A')
            
            risks_for_df.append(risk_row)

    if risks_for_df:
        df_risks_display = pd.DataFrame(risks_for_df)
        
        ws.cell(row=current_row_excel, column=1).value = "Input Risk Assumptions (from Input Data)"
        ws.cell(row=current_row_excel, column=1).font = title_font
        current_row_excel += 1
        df_risks_display.to_excel(writer, sheet_name=ws_name, startrow=current_row_excel -1, index=False)
        
        risk_col_styles = {}
        # Apply styles based on column content
        for idx, col_name in enumerate(df_risks_display.columns):
            if col_name == 'Probability':
                risk_col_styles[idx] = percent_style.name
            elif '(days)' in col_name and 'Dist. Type' not in col_name:
                risk_col_styles[idx] = integer_style.name
            elif '($)' in col_name and 'Dist. Type' not in col_name:
                risk_col_styles[idx] = currency_style.name
        
        _style_dataframe_output(ws, len(df_risks_display), len(df_risks_display.columns),
                                start_row=current_row_excel, start_col=1,
                                col_styles=risk_col_styles, workbook=wb,
                                auto_width_padding=2)  # Reduced padding due to many columns
    else:
        ws.cell(row=current_row_excel, column=1).value = "No risk data found in input_data['risks']."
        ws.cell(row=current_row_excel, column=1).font = title_font
        current_row_excel +=1


def _create_distribution_data_sheet(writer, simulation_results):
    """Creates the 'Distribution Data' sheet with sample durations and costs."""
    ws_name = "Distribution Data"; wb = writer.book
    if ws_name not in wb.sheetnames: ws = wb.create_sheet(title=ws_name)
    else: ws = wb[ws_name]
    writer.sheets[ws_name] = ws
    distribution_sample = simulation_results.get('distributionSample', {})
    durations = distribution_sample.get('durations', []); costs = distribution_sample.get('costs', [])
    max_len = max(len(durations), len(costs))
    durations_padded = durations + [None] * (max_len - len(durations))
    costs_padded = costs + [None] * (max_len - len(costs))
    if max_len > 0:
        df_distribution = pd.DataFrame({'Sample Durations': durations_padded, 'Sample Costs': costs_padded})
        df_distribution.to_excel(writer, sheet_name=ws_name, index=False, startrow=0)
        col_styles = {0: integer_style.name, 1: currency_style.name}
        _style_dataframe_output(ws, len(df_distribution), len(df_distribution.columns),
                                start_row=1, start_col=1, col_styles=col_styles, workbook=wb)
    else:
        ws['A1'] = "No distribution sample data available."; ws['A1'].font = Font(italic=True)


# --- Main Report Generation Function ---
def create_excel_report(input_data, simulation_results, file_path="Simulation_Report.xlsx"):
    """Generates a formatted multi-sheet Excel report."""
    try:
        processed_simulation_results = convert_numpy_types(simulation_results)
        processed_input_data = convert_numpy_types(input_data)
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            _add_named_styles(writer.book)
            if "Summary" not in writer.book.sheetnames: summary_ws = writer.book.create_sheet("Summary", 0)
            else: summary_ws = writer.book["Summary"]
            writer.sheets["Summary"] = summary_ws
            _create_summary_sheet(writer, processed_input_data, processed_simulation_results)
            _create_inputs_assumptions_sheet(writer, processed_input_data)
            _create_distribution_data_sheet(writer, processed_simulation_results)
            if "Summary" in writer.book.sheetnames: writer.book.active = writer.book.sheetnames.index("Summary")
        print(f"Excel report successfully generated: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error creating Excel report at {file_path}: {e}"); import traceback; traceback.print_exc()
        return None

# --- Test Block ---
if __name__ == "__main__":
    # UPDATED sample_input_data to match new structure
    sample_input_data = {
        "projectName": "Advanced CRM Rollout (Input Data)", # Key: projectName
        "teamSize": np.int64(8),
        "dailyCost": np.float64(550.00), # This is total daily project cost, not per person
        "phases": [
            {
                "phaseName": "Phase 1: Planning & Design",
                "tasks": [
                    {"id": "T1", "name": "Requirements Gathering", "best": 8, "likely": 10, "worst": 15, "predecessors": []},
                    {"id": "T2", "name": "System Architecture", "best": 10, "likely": 12, "worst": 20, "predecessors": ["T1"]},
                    {"id": "T3", "name": "UI/UX Design", "best": 12, "likely": 15, "worst": 22, "predecessors": ["T1"]},
                ]
            },
            {
                "phaseName": "Phase 2: Development",
                "tasks": [
                    {"id": "T4", "name": "Backend Development", "best": 30, "likely": 40, "worst": 60, "predecessors": ["T2"]},
                    {"id": "T5", "name": "Frontend Development", "best": 25, "likely": 35, "worst": 50, "predecessors": ["T3"]},
                ]
            }
        ],
        "risks": [
            {
                "id": "R1", "name": "Key Supplier Delay", "probability": 0.15,
                "impactDays": {"best": 10, "likely": 20, "worst": 30},
                "impactCost": {"best": 5000, "likely": 10000, "worst": 15000},
                "correlates": {"risk_id": "R3", "prob_increase": 0.25}
            },
            {
                "id": "R2", "name": "Lead Developer Leaves", "probability": 0.05,
                "impactDays": {"best": 20, "likely": 30, "worst": 45},
                "impactCost": {"best": 15000, "likely": 25000, "worst": 40000}
            },
            {
                "id": "R3", "name": "Integration Complexity Underestimated", "probability": 0.20, # Missing impactDays for testing robustness
                "impactCost": {"best": 8000, "likely": 12000, "worst": 20000}
            }
        ]
    }

    # UPDATED new_sample_simulation_results to include p10 and p90
    new_sample_simulation_results = {
        'projectName': 'Advanced CRM Rollout (Simulated)', 
        'simulationRuns': 1000,
        'durationAnalysis': {
            'mean': np.float64(36.29), 'p5': np.float64(24.36),
            'p10': np.float64(26.50), # ADDED P10
            'p50': np.float64(31.27), 
            'p80': np.float64(45.79), 
            'p90': np.float64(52.15), # ADDED P90
            'p95': np.float64(60.49)
        },
        'costAnalysis': {
            'mean': np.float64(164329.87), 'p5': np.float64(107188.98),
            'p10': np.float64(115300.50), # ADDED P10
            'p50': np.float64(140199.38), 
            'p80': np.float64(208475.48), 
            'p90': np.float64(245600.75), # ADDED P90
            'p95': np.float64(287691.20)
        },
        'topRiskDriversByDays': [
            {'id': 'R1', 'name': 'Key Supplier Delay', 'occurrences': np.int64(148), 'totalImpactDays': np.float64(3020.29)},
            {'id': 'R4', 'name': 'Integration Complexity', 'occurrences': np.int64(196), 'totalImpactDays': np.float64(2710.66)},
            {'id': 'R2', 'name': 'Lead Developer Leaves', 'occurrences': np.int64(45), 'totalImpactDays': np.float64(1578.06)}
        ],
        'topRiskDriversByCost': [
            {'id': 'R3', 'name': 'Budget Overrun', 'occurrences': np.int64(132), 'totalImpactCost': np.float64(2167960.83)},
            {'id': 'R1', 'name': 'Key Supplier Delay', 'occurrences': np.int64(148), 'totalImpactCost': np.float64(1494220.55)},
        ],
        'distributionSample': {
            'durations': np.array([28.51, 45.34, 33.8, 30.1, 55.9, 22.5, 68.2, 31.0, 39.5, 42.1,
                                   25.0, 60.0, 30.5, 35.5, 40.5, 50.5, 29.5, 33.3, 44.4, 55.5,
                                   27.7, 37.7, 47.7, 52.3, 61.8, 21.2, 41.2, 31.2, 38.9, 48.1], dtype=np.float64),
            'costs': np.array([139757.58, 209711.26, 150000.0, 145000.0, 250000.0, 100000.0, 300000.0,
                               148000.0, 180000.0, 195000.0, 120000.0, 280000.0, 146000.0, 165000.0,
                               185000.0, 225000.0, 140000.0, 155000.0, 205000.0, 260000.0,
                               130000.0, 175000.0, 215000.0, 240000.0, 290000.0, 95000.0, 190000.0,
                               147000.0, 178000.0, 220000.0], dtype=np.float64)
        }
    }

    report_file = create_excel_report(sample_input_data, new_sample_simulation_results, "Final_Touch_Simulation_Report.xlsx")
    if report_file:
        print(f"Test report generated: {report_file}")
    else:
        print("Test report generation FAILED.")

    # Test with minimal data, now including p10 and p90
    sample_input_data_minimal = {
        "projectName": "Minimal Project",
        "teamSize": 1,
        "dailyCost": 100,
        "phases": [], 
        "risks": []   
    }
    new_sample_simulation_results_minimal = {
        'projectName': 'Minimal Sim Project',
        'durationAnalysis': {'p10': 9.0, 'p50': 10.0, 'p90': 12.0, 'p95': 15.0}, # Added p10, p90
        'costAnalysis': {'p10': 900.0, 'p50': 1000.0, 'p90': 1200.0, 'p95': 1600.0}, # Added p10, p90
        'topRiskDriversByDays': [], 'topRiskDriversByCost': [],
        'distributionSample': {'durations': [8,9,10,10,11,12,10,11,14,15], 'costs': [800,900,1000,1000,1100,1200,1000,1100,1400,1500]}
    }
    report_file_minimal = create_excel_report(sample_input_data_minimal, new_sample_simulation_results_minimal, "Final_Touch_Simulation_Report_Minimal.xlsx")
    if report_file_minimal:
        print(f"Minimal test report generated: {report_file_minimal}")
    else:
        print("Minimal test report generation FAILED.")

