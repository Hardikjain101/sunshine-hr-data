"""
================================================================================
HR ATTENDANCE ANALYTICS DASHBOARD
================================================================================
A complete end-to-end data pipeline for HR attendance analysis with
interactive Streamlit dashboard for management decision-making.
Author: HR Analytics Team
Version: 1.0
Date: 2025
================================================================================
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
from typing import Tuple, Dict, List, Optional
import calendar
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Centralized configuration for business rules and thresholds"""
    
    # Persistent Storage
    DATA_FILE_PATH = "attendance_data.xlsx"
    
    # Department Auto-Mapping
    DEPARTMENT_MAPPING = {
        'Keyra': 'ex employees',
        'Brianna': 'Nurse practitioner',
        'Candice': 'Counselors',
        'Brenda': 'Mid Office',
        'Megan': 'Front Desk',
        'Heather': 'Mid Office',
        'Shelbie': 'ex employees',
        'Brittany': 'Mid Office',
        'Dasha': 'Front Desk',
        'Mhykeisha': 'Nurse practitioner',
        'Kenyelle': 'Mid Office',
        'Jasmine': 'Mid Office',
        'Courtney': 'ex employees',
        'Jazmine': 'Mid Office',
        'Breanne': 'Nurse practitioner',
        'Stacey': 'Nurse practitioner',
        'Allison': 'Nurse practitioner',
        'Jaime': 'Nurse practitioner',
        'Natalie': 'Front Desk',
        'Roshon': 'Mid Office',
        'Susan': 'Nurse practitioner'
    }
    
    # Business hours configuration
    STANDARD_START_TIME = time(8, 0)      # 8:00 AM
    LATE_GRACE_PERIOD_END = time(8, 8)    # 8:08 AM
    VERY_LATE_THRESHOLD = time(8, 15)     # 8:15 AM
    STANDARD_END_TIME = time(17, 0)       # 5:00 PM
    
    # Early Departure Times
    EARLY_DEPARTURE_TIME_MON_THU = time(16, 30) # 4:30 PM
    EARLY_DEPARTURE_TIME_FRI = time(12, 15)   # 12:15 PM
    
    # Friday Full Day Threshold
    FRIDAY_FULL_DAY_PUNCH_OUT = time(12, 15) # 12:15 PM
    
    # Working hours thresholds
    MIN_WORK_HOURS = 4.0                   # Minimum daily hours
    MAX_WORK_HOURS = 10.0                  # Maximum reasonable hours
    HALF_DAY_THRESHOLD = 5.0               # Hours for half-day
    FULL_DAY_THRESHOLD = 8.0               # Hours for full-day
    
    # Overtime thresholds
    WEEKLY_STANDARD_HOURS = 40.0           # Weekly standard hours
    MONTHLY_STANDARD_HOURS = 160.0         # Monthly standard hours
    
    # Anomaly detection
    DUPLICATE_THRESHOLD_MINUTES = 5        # Minutes to detect duplicates
    EXCESSIVE_SHIFT_HOURS = 10.0           # Flag excessive shifts
    SHORT_SHIFT_HOURS = 4.0                # Flag short shifts
    
    # Consistency scoring
    EXPECTED_WORKING_DAYS = 22             # Expected days per month

# ============================================================================
# PHASE 1: DATA CLEANING & STANDARDIZATION
# ============================================================================

class DataCleaner:
    """Handles all data cleaning and standardization operations"""
    
    @staticmethod
    def fill_missing_departments(df: pd.DataFrame) -> pd.DataFrame:
        """Auto-fill missing departments based on first name mapping"""
        if 'Department' not in df.columns:
            df['Department'] = np.nan
            
        def get_dept(row):
            curr = row.get('Department')
            if pd.notna(curr) and str(curr).strip() not in ['', 'nan', 'None', 'Unknown']:
                return curr
            fname = str(row.get('Employee First Name', '')).strip().title()
            return Config.DEPARTMENT_MAPPING.get(fname, 'Unknown')
            
        df['Department'] = df.apply(get_dept, axis=1)
        return df

    @staticmethod
    def standardize_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine and standardize employee names
        Creates: Employee Full Name with proper Title Case
        """
        # Handle missing middle names
        df['Employee Middle Name'] = df['Employee Middle Name'].fillna('')
        
        # Combine names with proper spacing
        def combine_name(row):
            first = str(row.get('Employee First Name', '')).strip()
            middle = str(row.get('Employee Middle Name', '')).strip()
            last = str(row.get('Employee Last Name', '')).strip()
            
            # Build full name
            parts = [first, middle, last] if middle else [first, last]
            full_name = ' '.join(filter(None, parts))
            
            # Apply Title Case and clean extra spaces
            return ' '.join(full_name.split()).title()
        
        df['Employee Full Name'] = df.apply(combine_name, axis=1)
        
        return df
    
    @staticmethod
    def clean_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert and standardize all datetime columns
        Creates authoritative timestamp and derived date/time fields
        """
        # Define datetime columns
        datetime_cols = {
            'Actual Date Time': 'actual_dt',
            'Punch Date Time': 'punch_dt',
            'Created Date Time (UTC)': 'created_dt'
        }
        
        # Convert to datetime
        for col, alias in datetime_cols.items():
            if col in df.columns:
                df[alias] = pd.to_datetime(df[col], errors='coerce')
        
        # Select authoritative timestamp (prefer Actual, fallback to Punch)
        df['Timestamp'] = df['actual_dt'].fillna(df['punch_dt'])
        
        # Create derived fields
        df['Punch Date'] = df['Timestamp'].dt.date
        df['Punch Time'] = df['Timestamp'].dt.time
        df['Day of Week'] = df['Timestamp'].dt.day_name()
        df['Week Number'] = df['Timestamp'].dt.isocalendar().week
        df['Month'] = df['Timestamp'].dt.month
        df['Month Name'] = df['Timestamp'].dt.strftime('%B')
        df['Year'] = df['Timestamp'].dt.year
        df['Hour'] = df['Timestamp'].dt.hour
        df['Date'] = pd.to_datetime(df['Punch Date'])
        
        return df
    
    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify duplicate punches within threshold timeframe
        Creates: Duplicate Punch flag
        """
        df = df.sort_values(['Employee Number', 'Timestamp'])
        df['Time_Diff_Minutes'] = df.groupby('Employee Number')['Timestamp'].diff().dt.total_seconds() / 60
        
        # Flag duplicates (within threshold)
        df['Duplicate Punch'] = (
            (df['Time_Diff_Minutes'] > 0) & 
            (df['Time_Diff_Minutes'] <= Config.DUPLICATE_THRESHOLD_MINUTES)
        )
        
        return df
    
    @staticmethod
    def create_data_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create flags for missing and invalid data
        """
        # Missing punch type
        if 'Type' in df.columns:
            df['Missing Punch Type'] = df['Type'].isna() | (df['Type'] == '')
        else:
            df['Missing Punch Type'] = False
        
        # Missing timestamp
        df['Missing Timestamp'] = df['Timestamp'].isna()
        
        # Missing employee info
        df['Incomplete Employee Record'] = (
            df['Employee Full Name'].isna() | 
            (df['Employee Full Name'] == '') |
            df['Employee Number'].isna()
        )
        
        return df
    
    @staticmethod
    def clean_system_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize system-related fields
        """
        # Normalize punch sources
        if 'Source' in df.columns:
            df['Source Normalized'] = df['Source'].fillna('Unknown').str.strip().str.title()
        
        # Clean IP addresses
        if 'IP Address' in df.columns:
            df['IP Address Clean'] = df['IP Address'].fillna('Unknown').str.strip()
        
        # Handle location and door
        if 'Location' in df.columns:
            df['Location'] = df['Location'].fillna('Not Specified')
        
        if 'Door' in df.columns:
            df['Door'] = df['Door'].fillna('Not Specified')
        
        return df

# ============================================================================
# PHASE 2: FEATURE ENGINEERING (BUSINESS LOGIC)
# ============================================================================

class FeatureEngineer:
    """Creates derived business metrics and features"""
    
    @staticmethod
    def calculate_daily_attendance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily attendance metrics per employee
        Returns: Daily summary DataFrame
        """
        # Filter valid records and exclude weekends
        valid_df = df[
            (~df['Missing Timestamp']) & 
            (~df['Incomplete Employee Record']) &
            (~df['Duplicate Punch']) &
            (~df['Day of Week'].isin(['Saturday', 'Sunday']))
        ].copy()
        
        # Group by employee and date
        daily_records = []
        
        for (emp_num, emp_name), group in valid_df.groupby(['Employee Number', 'Employee Full Name']):
            for date, date_group in group.groupby('Punch Date'):
                date_group = date_group.sort_values('Timestamp')
                
                # Get first and last punch
                first_punch = date_group['Timestamp'].min()
                last_punch = date_group['Timestamp'].max()
                
                # Calculate duration
                duration = (last_punch - first_punch).total_seconds() / 3600
                
                # Count punches (excluding meal breaks for work punch count)
                punch_count = len(date_group)
                
                # Filter Normal punches for determining valid punch pairs
                normal_punches = date_group[date_group['Type'] == 'Normal'] if 'Type' in date_group.columns else date_group
                normal_punch_count = len(normal_punches)
                
                # Determine valid punch-in/out based on actual punch pattern
                # Has valid punch-in: First punch exists (always true if punch_count > 0)
                has_punch_in = punch_count > 0
                
                # Has valid punch-out: 
                # - If multiple punches (punch_count > 1), there must be an exit (last punch is the exit)
                # - If single punch (punch_count == 1), check if first != last (shouldn't happen but handles edge case)
                # - In practice: multiple punches = has exit, single punch = missing exit
                has_punch_out = punch_count > 1 or (first_punch != last_punch)
                
                # Get additional info - handle missing values gracefully
                if 'Department' in date_group.columns and pd.notna(date_group['Department'].iloc[0]):
                    department = date_group['Department'].iloc[0]
                else:
                    department = 'Unknown'
                
                if 'Employee Supervisor' in date_group.columns and pd.notna(date_group['Employee Supervisor'].iloc[0]):
                    supervisor = date_group['Employee Supervisor'].iloc[0]
                else:
                    supervisor = 'Unknown'
                
                daily_records.append({
                    'Employee Number': emp_num,
                    'Employee Full Name': emp_name,
                    'Department': department,
                    'Supervisor': supervisor,
                    'Date': date,
                    'First Punch In': first_punch,
                    'Last Punch Out': last_punch,
                    'Working Hours': round(duration, 2),
                    'Punch Count': punch_count,
                    'Normal Punch Count': normal_punch_count,
                    'Has Punch In': has_punch_in,
                    'Has Punch Out': has_punch_out,
                    'Day of Week': first_punch.strftime('%A'),
                    'Week Number': first_punch.isocalendar()[1],
                    'Month': first_punch.strftime('%B'),
                    'Year': first_punch.year
                })
        
        daily_df = pd.DataFrame(daily_records)
        return daily_df
    
    @staticmethod
    def add_compliance_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add compliance-related flags and metrics based on updated business rules.
        """
        if daily_df.empty:
            return daily_df

        def apply_compliance_rules(row):
            # ===================
            # Late Arrival Logic
            # ===================
            punch_in_time = row['First Punch In'].time()
            is_late = punch_in_time > Config.LATE_GRACE_PERIOD_END
            is_very_late = punch_in_time > Config.VERY_LATE_THRESHOLD
            
            minutes_late = 0
            if is_late:
                minutes_late = max(0, (
                    datetime.combine(datetime.today(), punch_in_time) -
                    datetime.combine(datetime.today(), Config.STANDARD_START_TIME)
                ).total_seconds() / 60)

            # =======================
            # Early Departure Logic
            # =======================
            day_of_week = row['Day of Week']
            punch_out_time = row['Last Punch Out'].time()
            
            if day_of_week == 'Friday':
                early_departure_threshold = Config.EARLY_DEPARTURE_TIME_FRI
            else:
                early_departure_threshold = Config.EARLY_DEPARTURE_TIME_MON_THU

            is_early_departure = punch_out_time < early_departure_threshold
            
            minutes_early = 0
            if is_early_departure:
                minutes_early = max(0, (
                    datetime.combine(datetime.today(), early_departure_threshold) -
                    datetime.combine(datetime.today(), punch_out_time)
                ).total_seconds() / 60)

            # ========================
            # Shift Classification
            # ========================
            working_hours = row['Working Hours']
            shift_type = 'Short Shift' # Default
            if day_of_week == 'Friday':
                # Friday is a full day if they punch out after the designated time
                if punch_out_time >= Config.FRIDAY_FULL_DAY_PUNCH_OUT:
                    shift_type = 'Full Day'
                else:
                    shift_type = 'Short Shift' # Or could be another category if needed
            else:
                # Standard weekday logic
                if working_hours >= Config.FULL_DAY_THRESHOLD:
                    shift_type = 'Full Day'
                elif working_hours >= Config.HALF_DAY_THRESHOLD:
                    shift_type = 'Half Day'

            return pd.Series([
                is_late, is_very_late, minutes_late,
                is_early_departure, minutes_early,
                shift_type
            ])

        # Apply rules
        daily_df[[
            'Is Late', 'Is Very Late', 'Minutes Late',
            'Is Early Departure', 'Minutes Early',
            'Shift Type'
        ]] = daily_df.apply(apply_compliance_rules, axis=1)

        # ========================
        # Missing Punch Logic
        # ========================
        daily_df['Missing Punch Out'] = (
            (daily_df['First Punch In'] == daily_df['Last Punch Out']) &
            (daily_df['Punch Count'] == 1)
        )
        daily_df['Missing Punch In'] = ~daily_df['Has Punch In']
        daily_df['Odd Punch Count'] = daily_df['Punch Count'] % 2 != 0
        
        return daily_df
    
    @staticmethod
    def calculate_productivity_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate employee-level productivity metrics
        Based on actual worked hours from valid attendance days
        """
        # Ensure Department is not NaN for grouping (fill with 'Unknown' if missing)
        daily_df = daily_df.copy()
        daily_df['Department'] = daily_df['Department'].fillna('Unknown')
        
        # Filter to valid working days: must have valid punch-in and actual worked hours
        # Include days with missing punch-out for completeness, but flag them separately
        valid_working_days = daily_df[
            (daily_df['Has Punch In']) &  # Must have punch-in
            (daily_df['Working Hours'] > 0)  # Must have worked some hours (duration > 0)
        ].copy()
        
        # If filtering removes all records, use all days (shouldn't happen normally)
        if len(valid_working_days) == 0:
            valid_working_days = daily_df.copy()
        
        # Group by employee to calculate productivity metrics from valid working days
        emp_metrics = valid_working_days.groupby(['Employee Number', 'Employee Full Name', 'Department']).agg({
            'Working Hours': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'Is Late': 'sum',
            'Is Early Departure': 'sum',
            'Missing Punch Out': 'sum',
            'Date': 'nunique'
        }).reset_index()
        
        # Flatten column names
        emp_metrics.columns = [
            'Employee Number', 'Employee Full Name', 'Department',
            'Total Hours', 'Avg Daily Hours', 'Std Hours', 'Min Hours', 'Max Hours', 'Total Days',
            'Late Count', 'Early Departure Count', 'Missing Punch Out Count', 'Unique Dates'
        ]
        
        # Round numeric columns
        numeric_cols = ['Total Hours', 'Avg Daily Hours', 'Std Hours', 'Min Hours', 'Max Hours']
        emp_metrics[numeric_cols] = emp_metrics[numeric_cols].round(2)
        
        # Handle division by zero for scores - ensure Total Days is at least 1
        emp_metrics['Total Days'] = emp_metrics['Total Days'].fillna(0)
        emp_metrics['Total Days'] = emp_metrics['Total Days'].replace(0, 1)  # Avoid division by zero for scores
        
        # Ensure all employees from daily_df are included (left merge to preserve all employees)
        all_employees = daily_df[['Employee Number', 'Employee Full Name', 'Department']].drop_duplicates()
        
        # Merge with all employees, keeping metrics for employees with valid working days
        # Fill missing values with 0 for employees with no valid working days
        emp_metrics = all_employees.merge(
            emp_metrics,
            on=['Employee Number', 'Employee Full Name', 'Department'],
            how='left'
        )
        
        # Fill missing numeric columns with 0 (for employees with no valid working days)
        fill_cols = ['Total Hours', 'Avg Daily Hours', 'Std Hours', 'Min Hours', 'Max Hours', 
                     'Total Days', 'Late Count', 'Early Departure Count', 'Missing Punch Out Count',
                     'Unique Dates']
        emp_metrics[fill_cols] = emp_metrics[fill_cols].fillna(0)
        
        # Ensure Total Days is at least 1 for employees with data (to avoid division issues)
        emp_metrics.loc[emp_metrics['Total Hours'] > 0, 'Total Days'] = emp_metrics.loc[
            emp_metrics['Total Hours'] > 0, 'Total Days'
        ].clip(lower=1)
        
        return emp_metrics
    
    @staticmethod
    def detect_anomalies(daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag various anomalies in attendance data
        """
        # Short shifts
        daily_df['Unusually Short'] = daily_df['Working Hours'] < Config.SHORT_SHIFT_HOURS
        
        # Excessive shifts
        daily_df['Unusually Long'] = daily_df['Working Hours'] > Config.EXCESSIVE_SHIFT_HOURS
        
        # Combined anomaly flag
        daily_df['Has Anomaly'] = (
            daily_df['Unusually Short'] |
            daily_df['Unusually Long'] |
            daily_df['Missing Punch Out'] |
            daily_df['Odd Punch Count']
        )
        
        return daily_df

    @staticmethod
    def calculate_overtime_metrics(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate weekly and monthly overtime metrics.
        """
        if daily_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Ensure 'Date' is datetime
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])

        # Weekly Overtime
        weekly_df = daily_df.copy()
        weekly_df['Week'] = weekly_df['Date'].dt.isocalendar().week
        weekly_df['Year'] = weekly_df['Date'].dt.year
        
        weekly_hours = weekly_df.groupby(['Employee Full Name', 'Year', 'Week'])['Working Hours'].sum().reset_index()
        weekly_hours['Weekly Overtime'] = weekly_hours['Working Hours'] - Config.WEEKLY_STANDARD_HOURS
        weekly_hours['Weekly Overtime'] = weekly_hours['Weekly Overtime'].clip(lower=0)
        
        # Monthly Overtime
        monthly_df = daily_df.copy()
        monthly_df['Month'] = monthly_df['Date'].dt.month
        monthly_df['Year'] = monthly_df['Date'].dt.year

        monthly_hours = monthly_df.groupby(['Employee Full Name', 'Year', 'Month'])['Working Hours'].sum().reset_index()
        monthly_hours['Monthly Overtime'] = monthly_hours['Working Hours'] - Config.MONTHLY_STANDARD_HOURS
        monthly_hours['Monthly Overtime'] = monthly_hours['Monthly Overtime'].clip(lower=0)

        return weekly_hours, monthly_hours

def plot_overtime_charts(overtime_df: pd.DataFrame, time_period: str, top_n: int = 15):
    """
    Create bar chart for weekly or monthly overtime from a pre-filtered DataFrame.
    """
    if overtime_df.empty or f'{time_period.capitalize()} Overtime' not in overtime_df.columns:
        return None
    
    overtime_col = f'{time_period.capitalize()} Overtime'
    
    # Filter for entries with actual overtime and get the top N
    overtime_df = overtime_df[overtime_df[overtime_col] > 0]
    top_performers = overtime_df.nlargest(top_n, overtime_col)
    
    if top_performers.empty:
        return None

    fig = px.bar(
        top_performers,
        x=overtime_col,
        y='Employee Full Name',
        orientation='h',
        title=f'Top {top_n} Employees by {time_period.capitalize()} Overtime',
        labels={overtime_col: 'Overtime Hours', 'Employee Full Name': 'Employee'},
        color=overtime_col,
        color_continuous_scale='Plasma'
    )
    fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
    return fig


# ============================================================================
# PHASE 3: DATA PERSISTENCE
# ============================================================================

class DataManager:
    """Handles data persistence and file management"""

    @staticmethod
    def _get_excel_extension(name_or_path: str) -> str:
        return os.path.splitext(str(name_or_path or ""))[1].lower()

    @staticmethod
    def _get_excel_engine(ext: str) -> Optional[str]:
        if ext == ".xlsx":
            return "openpyxl"
        if ext == ".xls":
            return "xlrd"
        return None
    
    @staticmethod
    def merge_and_save(uploaded_file, target_path: str):
        """
        Merge uploaded data with existing data and save to disk
        """
        try:
            # Load new data
            uploaded_file.seek(0)
            upload_ext = DataManager._get_excel_extension(getattr(uploaded_file, "name", ""))
            if upload_ext not in [".xlsx", ".xls"]:
                st.error("Unsupported file type. Please upload a .xlsx or .xls file.")
                return False
            upload_engine = DataManager._get_excel_engine(upload_ext)
            try:
                new_df = pd.read_excel(uploaded_file, engine=upload_engine)
            except Exception:
                uploaded_file.seek(0)
                try:
                    new_df = pd.read_excel(uploaded_file)
                except Exception:
                    st.error("Error reading the uploaded Excel file. Please verify it is a valid .xlsx or .xls.")
                    return False
            
            # Check if target file exists
            if os.path.exists(target_path):
                try:
                    target_ext = DataManager._get_excel_extension(target_path)
                    target_engine = DataManager._get_excel_engine(target_ext)
                    existing_df = pd.read_excel(target_path, engine=target_engine)
                    # Combine datasets
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                    # Remove exact duplicates
                    combined_df = combined_df.drop_duplicates()
                except Exception:
                    combined_df = new_df
            else:
                combined_df = new_df
            
            # Save merged data (atomic write)
            base_path, target_ext = os.path.splitext(target_path)
            if target_ext.lower() not in [".xlsx", ".xls"]:
                st.error("Data file path must be .xlsx or .xls.")
                return False
            target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
            os.makedirs(target_dir, exist_ok=True)
            target_engine = DataManager._get_excel_engine(target_ext.lower())
            import tempfile
            with tempfile.NamedTemporaryFile(
                prefix=os.path.basename(base_path) + ".tmp.",
                suffix=target_ext,
                dir=target_dir,
                delete=False
            ) as tmp_file:
                temp_path = tmp_file.name
            with pd.ExcelWriter(temp_path, engine=target_engine) as writer:
                combined_df.to_excel(writer, index=False)
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temp file not created: {temp_path}")
            os.replace(temp_path, target_path)
            return True
            
        except Exception as e:
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            st.error(f"Error saving data: {str(e)}")
            return False

# ============================================================================
# DATA LOADING & PROCESSING PIPELINE
# ============================================================================

@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete data processing pipeline
    Returns: (raw_df, daily_df, employee_metrics_df, weekly_overtime_df, monthly_overtime_df)
    """
    try:
        # Load data
        raw_df = pd.read_excel(uploaded_file)
        
        # Phase 1: Data Cleaning
        cleaner = DataCleaner()
        raw_df = cleaner.standardize_names(raw_df)
        raw_df = cleaner.fill_missing_departments(raw_df)
        raw_df = cleaner.clean_datetime_columns(raw_df)
        raw_df = cleaner.detect_duplicates(raw_df)
        raw_df = cleaner.create_data_quality_flags(raw_df)
        raw_df = cleaner.clean_system_metadata(raw_df)
        
        # Phase 2: Feature Engineering
        engineer = FeatureEngineer()
        daily_df = engineer.calculate_daily_attendance(raw_df)
        if daily_df.empty:
            st.warning("No valid attendance data found in the provided file for weekdays.")
            return None, None, None, None, None

        daily_df = engineer.add_compliance_metrics(daily_df)
        daily_df = engineer.detect_anomalies(daily_df)
        
        # Employee-level metrics
        emp_metrics_df = engineer.calculate_productivity_metrics(daily_df)

        # Overtime metrics
        weekly_overtime_df, monthly_overtime_df = engineer.calculate_overtime_metrics(daily_df)
        
        return raw_df, daily_df, emp_metrics_df, weekly_overtime_df, monthly_overtime_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None

# ============================================================================
# CACHED AGGREGATIONS (PERFORMANCE)
# ============================================================================

@st.cache_data(show_spinner=False)
def get_productivity_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer.calculate_productivity_metrics(daily_df)

@st.cache_data(show_spinner=False)
def get_dow_summary(view_df: pd.DataFrame) -> pd.Series:
    return view_df.groupby('Day of Week')['Working Hours'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
    ])

@st.cache_data(show_spinner=False)
def get_monthly_metrics_cached(daily_df: pd.DataFrame) -> pd.DataFrame:
    return calculate_monthly_metrics(daily_df)

@st.cache_data(show_spinner=False)
def get_recent_changes(monthly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df_sorted = monthly_df.sort_values(['Employee Full Name', 'YearMonth'])
    monthly_df_sorted['Prev Total Hours'] = monthly_df_sorted.groupby('Employee Full Name')['Total Hours'].shift(1)
    monthly_df_sorted['Hours Change'] = monthly_df_sorted['Total Hours'] - monthly_df_sorted['Prev Total Hours']
    monthly_df_sorted['Hours Change %'] = (monthly_df_sorted['Hours Change'] / monthly_df_sorted['Prev Total Hours'] * 100).round(1)
    available_months = sorted(monthly_df_sorted['YearMonth'].unique().tolist())
    if not available_months:
        return monthly_df_sorted.iloc[0:0]
    recent_changes = monthly_df_sorted[monthly_df_sorted['YearMonth'] == available_months[-1]].copy()
    recent_changes = recent_changes[recent_changes['Prev Total Hours'].notna()].sort_values('Hours Change', ascending=False)
    return recent_changes

@st.cache_data(show_spinner=False)
def get_work_pattern_kpis_cached(
    daily_df: pd.DataFrame, employee_name: str, year: int, month: int
) -> Dict[str, float]:
    return calculate_work_pattern_kpis(daily_df, employee_name, year, month)

@st.cache_data(show_spinner=False)
def get_work_pattern_distribution_cached(
    daily_df: pd.DataFrame, employee_name: str, year: int, month: int
) -> pd.DataFrame:
    return calculate_work_pattern_distribution(daily_df, employee_name, year, month)

@st.cache_data(show_spinner=False)
def get_work_pattern_calendar_cached(
    daily_df: pd.DataFrame,
    employee_name: str,
    year: int,
    month: int,
    kpi_data: Optional[Dict[str, float]] = None
) -> str:
    return create_work_pattern_calendar(daily_df, employee_name, year, month, kpi_data)

@st.cache_data(show_spinner=False)
def count_working_days(start_date, end_date) -> int:
    if start_date is None or end_date is None:
        return 0
    if end_date < start_date:
        start_date, end_date = end_date, start_date
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")
    return int((date_index.weekday < 5).sum())

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_metric_card(label: str, value, delta=None, help_text=None):
    """Create a styled metric card"""
    col = st.container()
    with col:
        if delta:
            st.metric(label=label, value=value, delta=delta, help=help_text)
        else:
            st.metric(label=label, value=value, help=help_text)

def plot_compliance_trend(daily_df: pd.DataFrame):
    """Line chart of compliance metrics over time"""
    # Aggregate by date
    trend = daily_df.groupby('Date').agg({
        'Is Late': lambda x: (x.sum() / len(x) * 100),
        'Is Early Departure': lambda x: (x.sum() / len(x) * 100)
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend['Date'], y=trend['Is Late'], 
                             mode='lines+markers', name='Late Arrivals %',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=trend['Date'], y=trend['Is Early Departure'],
                             mode='lines+markers', name='Early Departures %',
                             line=dict(color='orange')))
    
    fig.update_layout(
        title='Compliance Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Percentage (%)',
        height=400,
        hovermode='x unified'
    )
    return fig

def plot_employee_ranking(emp_metrics_df: pd.DataFrame, metric: str, top_n: int = 10):
    """Horizontal bar chart for employee rankings"""
    top_emp = emp_metrics_df.nlargest(top_n, metric)
    
    fig = px.bar(
        top_emp,
        y='Employee Full Name',
        x=metric,
        orientation='h',
        title=f'Top {top_n} Employees by {metric}',
        labels={metric: metric, 'Employee Full Name': 'Employee'},
        color=metric,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

# ============================================================================
# MONTHLY ANALYTICS FUNCTIONS
# ============================================================================

def calculate_monthly_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly performance metrics per employee
    Returns DataFrame with monthly aggregations
    """
    # Ensure Date is datetime
    daily_df = daily_df.copy()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df['YearMonth'] = daily_df['Date'].dt.to_period('M').astype(str)
    
    # Calculate monthly metrics per employee
    monthly_metrics = daily_df.groupby(['Employee Number', 'Employee Full Name', 'Department', 'YearMonth']).agg({
        'Working Hours': ['sum', 'mean'],
        'Date': 'nunique',
        'Is Late': 'sum',
        'Is Early Departure': 'sum',
        'Missing Punch Out': 'sum'
    }).reset_index()
    
    # Flatten column names
    monthly_metrics.columns = [
        'Employee Number', 'Employee Full Name', 'Department', 'YearMonth',
        'Total Hours', 'Avg Daily Hours', 'Attendance Days',
        'Late Count', 'Early Departure Count', 'Missing Punch Out Count'
    ]
    
    # Round numeric columns
    monthly_metrics['Total Hours'] = monthly_metrics['Total Hours'].round(2)
    monthly_metrics['Avg Daily Hours'] = monthly_metrics['Avg Daily Hours'].round(2)
    
    # Sort by YearMonth for proper chronological order
    monthly_metrics = monthly_metrics.sort_values(['Employee Full Name', 'YearMonth'])
    
    return monthly_metrics

def plot_monthly_trend(monthly_df: pd.DataFrame, employee_name: str, metric: str = 'Total Hours'):
    """Line chart showing employee's monthly performance trend"""
    emp_data = monthly_df[monthly_df['Employee Full Name'] == employee_name].sort_values('YearMonth')
    
    if len(emp_data) == 0:
        return None
    
    fig = px.line(
        emp_data,
        x='YearMonth',
        y=metric,
        markers=True,
        title=f'{employee_name} - Monthly {metric} Trend',
        labels={'YearMonth': 'Month', metric: metric},
        line_shape='linear'
    )
    fig.update_traces(line_color='#2E86AB', marker_size=8)
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_monthly_comparison(monthly_df: pd.DataFrame, year_month: str, metric: str = 'Total Hours', top_n: int = 10):
    """Bar chart comparing employees for a specific month"""
    month_data = monthly_df[monthly_df['YearMonth'] == year_month].sort_values(metric, ascending=False).head(top_n)
    
    if len(month_data) == 0:
        return None
    
    fig = px.bar(
        month_data,
        x=metric,
        y='Employee Full Name',
        orientation='h',
        title=f'Top {top_n} Employees - {year_month} ({metric})',
        labels={metric: metric, 'Employee Full Name': 'Employee'},
        color=metric,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def calculate_attendance_distribution(daily_df: pd.DataFrame, employee_name: str, year: int, month: int) -> pd.DataFrame:
    """
    Calculate distribution of attendance types for an employee in a specific month
    Returns DataFrame with counts for each attendance type
    """
    # Filter to employee and month
    emp_df = daily_df[daily_df['Employee Full Name'] == employee_name].copy()
    emp_df['Date'] = pd.to_datetime(emp_df['Date'])
    emp_df = emp_df[(emp_df['Date'].dt.year == year) & (emp_df['Date'].dt.month == month)]
    
    # Categorize each day
    def categorize_day(row):
        # Use Shift Type from compliance logic to ensure Friday Full Days are counted correctly
        shift_type = row.get('Shift Type', 'Absent')
        has_anomaly = row.get('Has Anomaly', False)
        
        if has_anomaly:
            return 'Anomaly'
        elif shift_type == 'Full Day':
            return 'Full Day'
        elif shift_type == 'Half Day':
            return 'Half Day'
        elif shift_type == 'Short Shift':
            return 'Short Day'
        elif row['Working Hours'] > 0:
            return 'Short Day'
        else:
            return 'Absent'
    
    emp_df['Attendance Type'] = emp_df.apply(categorize_day, axis=1)
    
    # Count distribution
    distribution = emp_df['Attendance Type'].value_counts().reset_index()
    distribution.columns = ['Attendance Type', 'Count']
    
    # Ensure all categories are present (fill missing with 0)
    all_types = ['Full Day', 'Half Day', 'Short Day', 'Absent', 'Anomaly']
    for atype in all_types:
        if atype not in distribution['Attendance Type'].values:
            distribution = pd.concat([distribution, pd.DataFrame({'Attendance Type': [atype], 'Count': [0]})], ignore_index=True)
    
    # Sort by predefined order
    type_order = {atype: i for i, atype in enumerate(all_types)}
    distribution['Order'] = distribution['Attendance Type'].map(type_order)
    distribution = distribution.sort_values('Order').drop('Order', axis=1)
    
    return distribution

def create_attendance_calendar(daily_df: pd.DataFrame, employee_name: str, year: int, month: int):
    """
    Create a calendar view for employee attendance in a specific month, with updated business logic.
    """
    # Filter to employee and month
    emp_df = daily_df[daily_df['Employee Full Name'] == employee_name].copy()
    emp_df['Date'] = pd.to_datetime(emp_df['Date'])
    emp_df = emp_df[(emp_df['Date'].dt.year == year) & (emp_df['Date'].dt.month == month)]
    
    # Create calendar data structure
    cal = calendar.Calendar(firstweekday=6)  # Start with Sunday
    
    # Get all days in the month
    month_days = cal.monthdayscalendar(year, month)
    
    # Create date mapping
    date_status = {row['Date'].day: row for _, row in emp_df.iterrows()}
    
    # Build HTML calendar - Start with header
    month_name = calendar.month_name[month]
    html = f'<div style="font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px;">'
    html += f'<h3 style="text-align: center; color: #2E86AB; margin-bottom: 20px;">{month_name} {year} - {employee_name}</h3>'
    html += '<table style="width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
    html += '<thead><tr style="background-color: #2E86AB; color: white;">'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Sun</th>'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Mon</th>'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Tue</th>'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Wed</th>'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Thu</th>'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Fri</th>'
    html += '<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Sat</th>'
    html += '</tr></thead><tbody>'
    
    # Color mapping
    colors = {
        'full': '#4CAF50',      # Green
        'half': '#FFC107',      # Yellow
        'short': '#FF9800',     # Orange
        'absent': '#F44336',    # Red
        'anomaly': '#9C27B0',    # Purple
        'weekoff': '#e0e0e0'    # Grey
    }
    
    status_labels = {
        'full': 'Full Day',
        'half': 'Half Day',
        'short': 'Short Day',
        'absent': 'Absent',
        'anomaly': 'Anomaly',
        'weekoff': 'Week Off'
    }
    
    for week in month_days:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += '<td style="padding: 15px; border: 1px solid #ddd; background-color: #f5f5f5;"></td>'
                continue

            # Determine weekday (0=Mon, 6=Sun)
            current_date = datetime(year, month, day)
            weekday = current_date.weekday()

            # Handle Weekends
            if weekday == 5 or weekday == 6: # Saturday or Sunday
                html += f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: {colors["weekoff"]}; color: #333; font-weight: bold; min-width: 100px;">'
                html += f'<div style="font-size: 16px; font-weight: bold;">{day}</div>'
                html += f'<div style="font-size: 10px; margin-top: 3px;">{status_labels["weekoff"]}</div>'
                html += '</td>'
                continue

            # Handle Weekdays
            day_info = date_status.get(day)
            
            if day_info is not None:
                status = 'absent' # Default
                # Use Shift Type calculated in compliance logic
                shift_type = day_info.get('Shift Type', 'Absent')
                
                if day_info.get('Has Anomaly', False):
                    status = 'anomaly'
                elif shift_type == 'Full Day':
                    status = 'full'
                elif shift_type == 'Half Day':
                    status = 'half'
                elif shift_type == 'Short Shift':
                    status = 'short'
                
                hours = day_info['Working Hours']
                bg_color = colors.get(status, '#ffffff')
                label = status_labels.get(status, '')
                
                # Tooltip info
                info = f"Status: {shift_type} | Hours: {hours:.1f}h"
                if day_info.get('First Punch In'):
                    info += f" | In: {day_info['First Punch In'].strftime('%H:%M')}"
                if day_info.get('Last Punch Out'):
                    info += f" | Out: {day_info['Last Punch Out'].strftime('%H:%M')}"
                if day_info.get('Is Late', False):
                    info += " | Late"
                if day_info.get('Is Very Late', False):
                    info += " (Very Late)"
                if day_info.get('Is Early Departure', False):
                    info += " | Early Departure"
                
                info_escaped = info.replace('"', '&quot;')
                html += f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: {bg_color}; color: white; font-weight: bold; min-width: 100px;" title="{info_escaped}">'
                html += f'<div style="font-size: 16px; font-weight: bold;">{day}</div>'
                html += f'<div style="font-size: 10px; margin-top: 3px; opacity: 0.9;">{label}</div>'
                
                # In/Out times
                if pd.notna(day_info.get('First Punch In')) or pd.notna(day_info.get('Last Punch Out')):
                    html += '<div style="font-size: 9px; margin-top: 4px; line-height: 1.2; opacity: 0.85;">'
                    if pd.notna(day_info.get('First Punch In')):
                        html += f'<div>In: {day_info["First Punch In"].strftime("%H:%M")}</div>'
                    if pd.notna(day_info.get('Last Punch Out')):
                        html += f'<div>Out: {day_info["Last Punch Out"].strftime("%H:%M")}</div>'
                    html += '</div>'
                
                html += '</td>'
            else:
                # Day with no punches (absent)
                bg_color = colors['absent']
                label = status_labels['absent']
                html += f'<td style="padding: 10px; text-align: center; border: 1px solid #ddd; background-color: {bg_color}; color: white; font-weight: bold; min-width: 100px;" title="Absent">'
                html += f'<div style="font-size: 16px; font-weight: bold;">{day}</div>'
                html += f'<div style="font-size: 10px; margin-top: 3px; opacity: 0.9;">{label}</div>'
                html += '</td>'

        html += "</tr>"
    
    html += '</tbody></table>'
    html += '<div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">'
    html += '<div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">'
    html += '<div style="display: flex; align-items: center;"><div style="width: 30px; height: 20px; background-color: #4CAF50; margin-right: 8px; border: 1px solid #ddd;"></div><span>Full Day</span></div>'
    html += '<div style="display: flex; align-items: center;"><div style="width: 30px; height: 20px; background-color: #FFC107; margin-right: 8px; border: 1px solid #ddd;"></div><span>Half Day</span></div>'
    html += '<div style="display: flex; align-items: center;"><div style="width: 30px; height: 20px; background-color: #FF9800; margin-right: 8px; border: 1px solid #ddd;"></div><span>Short Day</span></div>'
    html += '<div style="display: flex; align-items.center;"><div style="width: 30px; height: 20px; background-color: #F44336; margin-right: 8px; border: 1px solid #ddd;"></div><span>Absent / No Punch</span></div>'
    html += '<div style="display: flex; align-items: center;"><div style="width: 30px; height: 20px; background-color: #9C27B0; margin-right: 8px; border: 1px solid #ddd;"></div><span>Anomaly</span></div>'
    html += '<div style="display: flex; align-items: center;"><div style="width: 30px; height: 20px; background-color: #e0e0e0; margin-right: 8px; border: 1px solid #ddd;"></div><span>Week Off</span></div>'
    html += '</div></div></div>'
    
    return html

def get_employee_work_pattern(employee_name: str):
    """
    Return expected workdays and optional early departure override for an employee.
    """
    first_name = str(employee_name).strip().split()[0].title() if employee_name else ''
    default_workdays = {0, 1, 2, 3, 4}
    work_patterns = {
        'Jaime': {'workdays': {0, 3}, 'early_departure': time(15, 0)},
        'Susan': {'workdays': {0, 4}},
        'Breanne': {'workdays': {1, 2, 3, 4}},
        'Mhykeisha': {'workdays': {0, 1, 2, 3}},
        'Candice': {'workdays': {1, 2, 3}}
    }
    pattern = work_patterns.get(first_name, {'workdays': default_workdays})
    return pattern['workdays'], pattern.get('early_departure')

def calculate_work_pattern_summary(daily_df: pd.DataFrame, employee_name: str, year: int, month: int) -> Dict[str, int]:
    """
    Summarize attendance counts using employee-specific work patterns.
    """
    emp_df = daily_df[daily_df['Employee Full Name'] == employee_name].copy()
    emp_df['Date'] = pd.to_datetime(emp_df['Date'])
    emp_df = emp_df[(emp_df['Date'].dt.year == year) & (emp_df['Date'].dt.month == month)]
    
    expected_workdays, _ = get_employee_work_pattern(employee_name)
    date_status = {row['Date'].day: row for _, row in emp_df.iterrows()}
    days_in_month = calendar.monthrange(year, month)[1]
    
    summary = {
        'total_days': days_in_month,
        'full_days': 0,
        'half_days': 0,
        'short_days': 0,
        'absent_days': 0,
        'week_off_days': 0,
        'worked_non_working_days': 0
    }
    
    for day in range(1, days_in_month + 1):
        current_date = datetime(year, month, day)
        weekday = current_date.weekday()
        is_weekend = weekday >= 5
        is_expected_workday = (weekday in expected_workdays) and not is_weekend
        day_info = date_status.get(day)
        
        if day_info is not None:
            if not is_expected_workday:
                summary['worked_non_working_days'] += 1
                continue
            
            shift_type = day_info.get('Shift Type', 'Absent')
            if pd.isna(shift_type):
                shift_type = 'Absent'
            
            if shift_type == 'Full Day':
                summary['full_days'] += 1
            elif shift_type == 'Half Day':
                summary['half_days'] += 1
            elif shift_type == 'Short Shift' or day_info.get('Working Hours', 0) > 0:
                summary['short_days'] += 1
            else:
                summary['absent_days'] += 1
        else:
            if is_expected_workday:
                summary['absent_days'] += 1
            else:
                summary['week_off_days'] += 1
    
    return summary

def calculate_work_pattern_distribution(daily_df: pd.DataFrame, employee_name: str, year: int, month: int) -> pd.DataFrame:
    """
    Build a distribution DataFrame based on work pattern summary.
    """
    summary = calculate_work_pattern_summary(daily_df, employee_name, year, month)
    distribution = [
        {'Attendance Type': 'Full Day', 'Count': summary['full_days']},
        {'Attendance Type': 'Half Day', 'Count': summary['half_days']},
        {'Attendance Type': 'Short Day', 'Count': summary['short_days']},
        {'Attendance Type': 'Absent', 'Count': summary['absent_days']},
        {'Attendance Type': 'Week Off', 'Count': summary['week_off_days']},
        {'Attendance Type': 'Worked on Non-Working Day', 'Count': summary['worked_non_working_days']}
    ]
    return pd.DataFrame(distribution)

def get_work_pattern_context_text(employee_name: str) -> str:
    """
    Build a short, human-readable message for custom work patterns.
    """
    expected_workdays, early_departure_override = get_employee_work_pattern(employee_name)
    default_workdays = {0, 1, 2, 3, 4}
    if expected_workdays == default_workdays and not early_departure_override:
        return ""

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    workday_names = [day_names[idx] for idx in sorted(expected_workdays)]

    if len(workday_names) == 1:
        days_text = workday_names[0]
    elif len(workday_names) == 2:
        days_text = f"{workday_names[0]} and {workday_names[1]}"
    else:
        days_text = f"{', '.join(workday_names[:-1])}, and {workday_names[-1]}"

    first_name = str(employee_name).strip().split()[0].title() if employee_name else "This employee"
    message = f"{first_name} works only on {days_text}."

    if early_departure_override:
        time_text = early_departure_override.strftime('%I:%M %p').lstrip('0')
        message += f" Early departure threshold: {time_text}."

    return message

def get_expected_daily_hours(weekday: int, early_departure_override=None) -> float:
    """
    Return expected daily hours based on weekday rules or an override.
    """
    if weekday >= 5:
        return 0.0

    if early_departure_override:
        end_time = early_departure_override
    else:
        end_time = Config.EARLY_DEPARTURE_TIME_FRI if weekday == 4 else Config.EARLY_DEPARTURE_TIME_MON_THU

    start_time = Config.STANDARD_START_TIME
    expected_hours = (
        datetime.combine(datetime.today(), end_time) -
        datetime.combine(datetime.today(), start_time)
    ).total_seconds() / 3600
    return max(0.0, expected_hours)

def calculate_work_pattern_kpis(daily_df: pd.DataFrame, employee_name: str, year: int, month: int) -> Dict[str, float]:
    """
    Calculate expected vs actual and punctuality KPIs for the work pattern calendar.
    """
    emp_df = daily_df[daily_df['Employee Full Name'] == employee_name].copy()
    emp_df['Date'] = pd.to_datetime(emp_df['Date'])
    emp_df = emp_df[(emp_df['Date'].dt.year == year) & (emp_df['Date'].dt.month == month)]
    actual_hours_all_days = emp_df['Working Hours'].sum() if not emp_df.empty else 0.0

    expected_workdays, early_departure_override = get_employee_work_pattern(employee_name)
    days_in_month = calendar.monthrange(year, month)[1]

    expected_dates = []
    expected_hours = 0.0
    for day in range(1, days_in_month + 1):
        current_date = datetime(year, month, day)
        weekday = current_date.weekday()
        is_weekend = weekday >= 5
        is_expected = (weekday in expected_workdays) and not is_weekend
        if is_expected:
            expected_dates.append(current_date.date())
            expected_hours += get_expected_daily_hours(weekday, early_departure_override)

    expected_days = len(expected_dates)

    if emp_df.empty:
        expected_df = emp_df
        non_working_df = emp_df
    else:
        emp_df['DateOnly'] = emp_df['Date'].dt.date
        expected_df = emp_df[emp_df['DateOnly'].isin(expected_dates)]
        non_working_df = emp_df[~emp_df['DateOnly'].isin(expected_dates)]

    actual_days = len(expected_df)
    actual_hours = expected_df['Working Hours'].sum() if not expected_df.empty else 0.0
    missed_days = max(0, expected_days - actual_days)

    if not expected_df.empty:
        late_series = expected_df['Is Late'].fillna(False)
        if early_departure_override:
            early_series = expected_df['Last Punch Out'].apply(
                lambda x: pd.notna(x) and x.time() < early_departure_override
            )
        else:
            early_series = expected_df['Is Early Departure'].fillna(False)

        early_series = early_series.fillna(False)
        late_count = int(late_series.sum())
        early_count = int(early_series.sum())
        on_time_days = int((~late_series & ~early_series).sum())
    else:
        late_count = 0
        early_count = 0
        on_time_days = 0

    return {
        'expected_days': expected_days,
        'actual_days': actual_days,
        'missed_days': missed_days,
        'expected_hours': expected_hours,
        'actual_hours': actual_hours,
        'actual_hours_all_days': actual_hours_all_days,
        'hours_diff': actual_hours - expected_hours,
        'late_arrivals': late_count,
        'early_departures': early_count,
        'on_time_days': on_time_days,
        'worked_non_working_days': len(non_working_df)
    }

def create_work_pattern_calendar(
    daily_df: pd.DataFrame,
    employee_name: str,
    year: int,
    month: int,
    kpi_data: Optional[Dict[str, float]] = None
):
    """
    Create a calendar view for employee attendance with employee-specific work patterns.
    """
    # Filter to employee and month
    emp_df = daily_df[daily_df['Employee Full Name'] == employee_name].copy()
    emp_df['Date'] = pd.to_datetime(emp_df['Date'])
    emp_df = emp_df[(emp_df['Date'].dt.year == year) & (emp_df['Date'].dt.month == month)]
    
    # Employee-specific work patterns (weekday: 0=Mon, 6=Sun)
    expected_workdays, early_departure_override = get_employee_work_pattern(employee_name)
    
    # Create calendar data structure
    cal = calendar.Calendar(firstweekday=6)  # Start with Sunday
    
    # Get all days in the month
    month_days = cal.monthdayscalendar(year, month)
    
    # Create date mapping
    date_status = {row['Date'].day: row for _, row in emp_df.iterrows()}

    if kpi_data is None:
        kpi_data = calculate_work_pattern_kpis(daily_df, employee_name, year, month)

    expected_days = int(kpi_data.get('expected_days', 0) or 0)
    actual_days = int(kpi_data.get('actual_days', 0) or 0)
    missed_days = int(kpi_data.get('missed_days', 0) or 0)
    expected_hours = float(kpi_data.get('expected_hours', 0.0) or 0.0)
    actual_hours = float(kpi_data.get('actual_hours', 0.0) or 0.0)
    hours_diff = float(kpi_data.get('hours_diff', 0.0) or 0.0)
    on_time_days = int(kpi_data.get('on_time_days', 0) or 0)
    late_arrivals = int(kpi_data.get('late_arrivals', 0) or 0)
    early_departures = int(kpi_data.get('early_departures', 0) or 0)
    worked_non_working_days = int(kpi_data.get('worked_non_working_days', 0) or 0)
    anomaly_days = int(emp_df['Has Anomaly'].sum()) if 'Has Anomaly' in emp_df.columns else 0

    attendance_rate = round((actual_days / expected_days) * 100) if expected_days else 0
    punctuality_rate = round((on_time_days / expected_days) * 100) if expected_days else 0

    def rate_class(value: float, good: float, warn: float) -> str:
        if value >= good:
            return 'good'
        if value >= warn:
            return 'warn'
        return 'bad'

    if expected_days == 0:
        attendance_class = punctuality_class = hours_class = exceptions_class = 'neutral'
        overall_class = 'neutral'
        overall_label = 'No Expected Days'
        overall_score_text = 'NA'
        status_note = 'No scheduled workdays in this month.'
    else:
        attendance_class = rate_class(attendance_rate, 90, 75)
        punctuality_class = rate_class(punctuality_rate, 85, 70)

        abs_hours_diff = abs(hours_diff)
        if abs_hours_diff <= 4:
            hours_class = 'good'
        elif abs_hours_diff <= 10:
            hours_class = 'warn'
        else:
            hours_class = 'bad'

        if missed_days == 0 and anomaly_days == 0 and worked_non_working_days == 0:
            exceptions_class = 'good'
        elif missed_days <= 2 and anomaly_days <= 1:
            exceptions_class = 'warn'
        else:
            exceptions_class = 'bad'

        overall_score = attendance_rate * 0.6 + punctuality_rate * 0.4
        overall_class = rate_class(overall_score, 85, 70)
        overall_label = {'good': 'On Track', 'warn': 'Needs Attention', 'bad': 'At Risk'}[overall_class]
        overall_score_text = f"{round(overall_score)}%"

        if missed_days > 0:
            status_note = f"{missed_days} expected day(s) missed."
        elif late_arrivals > 0 or early_departures > 0:
            status_note = "Timing flags present."
        else:
            status_note = "Attendance and timing look steady."
    
    # Build HTML calendar - Start with header
    month_name = calendar.month_name[month]
    hours_balance_text = f"{hours_diff:+.1f}h"
    if expected_days:
        hours_balance_note = f"{actual_hours:.1f}h actual / {expected_hours:.1f}h expected"
    else:
        hours_balance_note = f"{actual_hours:.1f}h logged"
    attendance_value = f"{attendance_rate}%" if expected_days else "NA"
    punctuality_value = f"{punctuality_rate}%" if expected_days else "NA"

    html = '<div class="cal-wrap">'
    html += """
        <style>
        .cal-wrap{font-family:Arial,sans-serif;max-width:980px;margin:0 auto;padding:16px;background:linear-gradient(180deg,#f5f8fb 0%,#fff 60%);border:1px solid #dde6ef;border-radius:12px;box-shadow:0 6px 18px rgba(22,43,60,.08);}
        .cal-header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;}
        .cal-title{font-size:20px;font-weight:700;color:#1f5f7a;}
        .cal-subtitle{font-size:12px;color:#5d6c79;}
        .status-summary{background:#fff;border:1px solid #dde6ef;border-radius:10px;padding:8px 10px;min-width:220px;}
        .status-label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#5d6c79;}
        .status-value{font-size:20px;font-weight:700;margin-top:4px;}
        .status-chip{display:inline-block;margin-top:6px;padding:4px 8px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;}
        .status-chip.good{background:#e8f6ee;color:#1e5d2a;border:1px solid #bfe4c9;}
        .status-chip.warn{background:#fff7dd;color:#7a5d00;border:1px solid #f3dd97;}
        .status-chip.bad{background:#ffe7e7;color:#8a1f1f;border:1px solid #f1b5b5;}
        .status-chip.neutral{background:#eef2f6;color:#4b5b66;border:1px solid #d5dde6;}
        .status-note{font-size:12px;color:#5d6c79;margin-top:6px;}
        .insight-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:12px;}
        .insight-card{background:#fff;border:1px solid #dde6ef;border-radius:10px;padding:10px;}
        .insight-card.good{border-left:4px solid #2f9e44;}
        .insight-card.warn{border-left:4px solid #f2c94c;}
        .insight-card.bad{border-left:4px solid #eb5757;}
        .insight-card.neutral{border-left:4px solid #cbd5df;}
        .insight-label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#5d6c79;}
        .insight-value{font-size:18px;font-weight:700;margin-top:4px;}
        .insight-sub,.insight-foot{font-size:12px;color:#5d6c79;margin-top:2px;}
        .cal-table{width:100%;border-collapse:separate;border-spacing:6px;margin-top:14px;table-layout:fixed;}
        .cal-table th{background:#2E86AB;color:#fff;padding:8px;font-size:11px;letter-spacing:.06em;text-transform:uppercase;border-radius:8px;}
        .cell-body{display:flex;flex-direction:column;height:100%;padding-bottom:16px;}
        .day-top{display:flex;justify-content:space-between;align-items:center;gap:6px;min-height:18px;}
        .day-num{font-size:14px;font-weight:700;}
        .status-pill{display:inline-block;padding:2px 6px;border-radius:999px;font-size:9px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#fff;}
        .hours-pill{display:inline-block;padding:2px 6px;border-radius:6px;font-size:10px;font-weight:700;background:#fff;border:1px solid #dde6ef;}
        .badge-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;min-height:18px;}
        .badge{font-size:9px;font-weight:700;padding:2px 5px;border-radius:4px;border:1px solid transparent;background:#fff;color:#1a1a1a;}
        .badge-late{border-color:#eb5757;color:#8a1f1f;}
        .badge-vlate{border-color:#b42323;background:#ffe1e1;color:#7b1515;}
        .badge-early{border-color:#f2994a;color:#8a4a00;}
        .badge-miss{border-color:#6c757d;color:#3f4a54;}
        .badge-anom{border-color:#6c4ab6;color:#3d2a6d;}
        .ot-pill{font-size:9px;font-weight:700;padding:2px 6px;border-radius:6px;background:#edf0f3;border:1px dashed #c7d1dc;color:#2f3a43;}
        .time-range{font-size:10px;color:#4b5b66;margin-top:4px;min-height:12px;}
        .cell-bar{position:absolute;left:8px;right:8px;bottom:8px;}
        .hours-track{margin-top:6px;height:6px;background:#e5ebf1;border-radius:999px;overflow:hidden;}
        .hours-track.placeholder{opacity:0;}
        .hours-fill{height:100%;display:block;background:#2f9e44;}
        .hours-fill.warn{background:#f2c94c;}
        .hours-fill.bad{background:#eb5757;}
        .hours-fill.off{background:#2e86ab;}
        .hours-fill.zero{background:#c8d0d9;}
        .off-tag{position:absolute;top:6px;right:6px;font-size:9px;font-weight:700;padding:2px 5px;border-radius:6px;background:#fff;border:1px solid #607d8b;color:#42535e;}
        .legend{margin-top:14px;padding:10px;background:#f7f9fb;border:1px solid #dde6ef;border-radius:10px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;}
        .legend-title{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#5d6c79;margin-bottom:6px;}
        .legend-item{display:flex;align-items:center;gap:6px;font-size:12px;color:#1f2933;margin-bottom:4px;}
        .legend-swatch{width:14px;height:14px;border-radius:4px;border:1px solid #dde6ef;background:#fff;}
        .legend-bar{width:24px;height:6px;border-radius:999px;background:#e5ebf1;position:relative;overflow:hidden;}
        .legend-bar:after{content:'';position:absolute;left:0;top:0;height:100%;width:60%;background:#2f9e44;}
        @media (max-width:900px){.cal-wrap{padding:10px}.cal-table{border-spacing:4px}.cal-title{font-size:18px}}
        </style>
    """
    html += (
        f'<div class="cal-header"><div><div class="cal-title">{month_name} {year}</div>'
        f'<div class="cal-subtitle">{employee_name}</div></div>'
        f'<div class="status-summary"><div class="status-label">Expectation Fit</div>'
        f'<div class="status-value">{overall_score_text}</div>'
        f'<div class="status-chip {overall_class}">{overall_label}</div>'
        f'<div class="status-note">{status_note}</div></div></div>'
    )
    html += '<div class="insight-row">'
    html += (
        f'<div class="insight-card {attendance_class}"><div class="insight-label">Attendance</div>'
        f'<div class="insight-value">{attendance_value}</div>'
        f'<div class="insight-sub">{actual_days} of {expected_days} expected days</div>'
        f'<div class="insight-foot">{missed_days} missed day(s)</div></div>'
    )
    html += (
        f'<div class="insight-card {punctuality_class}"><div class="insight-label">Punctuality</div>'
        f'<div class="insight-value">{punctuality_value}</div>'
        f'<div class="insight-sub">On time {on_time_days} of {expected_days} days</div>'
        f'<div class="insight-foot">Late {late_arrivals} | Early {early_departures}</div></div>'
    )
    html += (
        f'<div class="insight-card {hours_class}"><div class="insight-label">Hours Balance</div>'
        f'<div class="insight-value">{hours_balance_text}</div>'
        f'<div class="insight-sub">{hours_balance_note}</div>'
        f'<div class="insight-foot">Gap vs expectation</div></div>'
    )
    html += (
        f'<div class="insight-card {exceptions_class}"><div class="insight-label">Exceptions</div>'
        f'<div class="insight-value">{missed_days} missed</div>'
        f'<div class="insight-sub">Off-day work {worked_non_working_days}</div>'
        f'<div class="insight-foot">Anomaly days {anomaly_days}</div></div>'
    )
    html += '</div>'
    html += '<table class="cal-table"><thead><tr>'
    html += '<th>Sun</th><th>Mon</th><th>Tue</th><th>Wed</th><th>Thu</th><th>Fri</th><th>Sat</th>'
    html += '</tr></thead><tbody>'
    
    # Color mapping
    colors = {
        'full': '#e9f7ee',
        'half': '#fff7dd',
        'short': '#fff0e1',
        'absent': '#ffe7e7',
        'anomaly': '#f4eeff',
        'weekoff': '#edf0f3'
    }
    pill_colors = {
        'full': '#2f9e44',
        'half': '#caa531',
        'short': '#cf6d21',
        'absent': '#b73b3b',
        'anomaly': '#5a3fa0',
        'weekoff': '#7a8794'
    }
    text_colors = {
        'weekoff': '#4b5b66'
    }
    
    status_labels = {
        'full': 'Full Day',
        'half': 'Half Day',
        'short': 'Short Day',
        'absent': 'Absent',
        'anomaly': 'Anomaly',
        'weekoff': 'Week Off'
    }
    
    non_working_border = '#607d8b'
    
    for week in month_days:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += '<td style="padding: 15px; border: 1px solid #ddd; background-color: #f5f5f5;"></td>'
                continue

            # Determine weekday (0=Mon, 6=Sun)
            current_date = datetime(year, month, day)
            weekday = current_date.weekday()
            is_weekend = weekday >= 5
            is_expected_workday = (weekday in expected_workdays) and not is_weekend

            day_info = date_status.get(day)
            expected_hours_day = get_expected_daily_hours(weekday, early_departure_override) if is_expected_workday else 0.0
            
            if day_info is not None:
                status = 'absent'  # Default
                shift_type = day_info.get('Shift Type', 'Absent')
                if pd.isna(shift_type):
                    shift_type = 'Absent'
                
                if day_info.get('Has Anomaly', False):
                    status = 'anomaly'
                elif shift_type == 'Full Day':
                    status = 'full'
                elif shift_type == 'Half Day':
                    status = 'half'
                elif shift_type == 'Short Shift':
                    status = 'short'
                elif day_info.get('Working Hours', 0) > 0:
                    status = 'short'
                
                worked_on_non_working = not is_expected_workday
                hours = float(day_info.get('Working Hours', 0.0) or 0.0)
                if weekday == 4:
                    daily_overtime = max(0.0, hours - 4.25)
                else:
                    daily_overtime = max(0.0, hours - 8.5)
                bg_color = colors.get(status, '#ffffff')
                pill_color = pill_colors.get(status, '#2E86AB')
                text_color = text_colors.get(status, '#1a1a1a')
                label = status_labels.get(status, '')

                # Tooltip info
                info = f"Status: {shift_type} | Hours: {hours:.1f}h"
                if expected_hours_day > 0:
                    info += f" | Expected: {expected_hours_day:.1f}h"
                if worked_on_non_working:
                    info = f"Worked on Non-Working Day | {info}"
                if pd.notna(day_info.get('First Punch In')):
                    info += f" | In: {day_info['First Punch In'].strftime('%H:%M')}"
                if pd.notna(day_info.get('Last Punch Out')):
                    info += f" | Out: {day_info['Last Punch Out'].strftime('%H:%M')}"
                if day_info.get('Is Late', False):
                    info += " | Late"
                if day_info.get('Is Very Late', False):
                    info += " (Very Late)"

                is_early_departure = day_info.get('Is Early Departure', False)
                if early_departure_override and pd.notna(day_info.get('Last Punch Out')):
                    is_early_departure = day_info['Last Punch Out'].time() < early_departure_override
                if is_early_departure:
                    info += " | Early Departure"
                    if early_departure_override:
                        info += f" ({early_departure_override.strftime('%H:%M')})"
                if day_info.get('Missing Punch Out', False):
                    info += " | Missing Punch Out"

                info_escaped = info.replace('"', '&quot;')
                cell_style = (
                    f"padding: 8px; border: 1px solid #d7dee7; border-radius: 10px; "
                    f"background-color: {bg_color}; color: {text_color}; min-height: 110px; "
                    "vertical-align: top; position: relative;"
                )
                if worked_on_non_working:
                    cell_style += f" box-shadow: inset 0 0 0 2px {non_working_border};"

                badges = []
                if day_info.get('Is Very Late', False):
                    badges.append('<span class="badge badge-vlate" title="Very Late">VL</span>')
                elif day_info.get('Is Late', False):
                    badges.append('<span class="badge badge-late" title="Late">L</span>')
                if is_early_departure:
                    badges.append('<span class="badge badge-early" title="Early Departure">E</span>')
                if day_info.get('Missing Punch Out', False):
                    badges.append('<span class="badge badge-miss" title="Missing Punch Out">M</span>')
                if day_info.get('Has Anomaly', False):
                    badges.append('<span class="badge badge-anom" title="Anomaly">A</span>')

                in_time = day_info['First Punch In'].strftime('%H:%M') if pd.notna(day_info.get('First Punch In')) else '--'
                out_time = day_info['Last Punch Out'].strftime('%H:%M') if pd.notna(day_info.get('Last Punch Out')) else '--'
                time_range = f"{in_time} - {out_time}" if in_time != '--' or out_time != '--' else ''

                if expected_hours_day > 0:
                    ratio = max(0.0, min(hours / expected_hours_day, 1.0))
                    if ratio >= 0.95:
                        bar_class = 'good'
                    elif ratio >= 0.7:
                        bar_class = 'warn'
                    else:
                        bar_class = 'bad'
                else:
                    ratio = 1.0 if hours > 0 else 0.0
                    bar_class = 'off' if hours > 0 else 'zero'

                html += f'<td class="cal-cell" style="{cell_style}" title="{info_escaped}">'
                html += '<div class="cell-body">'
                html += '<div class="day-top">'
                html += f'<div class="day-num">{day}</div>'
                html += f'<span class="status-pill" style="background-color: {pill_color};">{label}</span>'
                html += '</div>'
                if worked_on_non_working:
                    html += '<div class="off-tag">OFF</div>'
                html += '<div class="badge-row">'
                html += f'<span class="hours-pill">{hours:.1f}h</span>'
                if daily_overtime > 0:
                    html += f'<span class="ot-pill">OT: +{daily_overtime:.1f}h</span>'
                if badges:
                    html += ''.join(badges)
                html += '</div>'
                html += f'<div class="time-range">{time_range or "&nbsp;"}</div>'
                html += '<div class="cell-bar">'
                if is_expected_workday or hours > 0:
                    html += (
                        f'<div class="hours-track"><span class="hours-fill {bar_class}" '
                        f'style="width: {ratio * 100:.0f}%;"></span></div>'
                    )
                else:
                    html += '<div class="hours-track placeholder"><span class="hours-fill zero" style="width: 0%;"></span></div>'
                html += '</div>'
                html += '</div>'
                html += '</td>'
            else:
                if is_expected_workday:
                    # Expected workday with no punches (absent)
                    bg_color = colors['absent']
                    pill_color = pill_colors['absent']
                    cell_style = (
                        f"padding: 8px; border: 1px solid #d7dee7; border-radius: 10px; "
                        f"background-color: {bg_color}; color: #1a1a1a; min-height: 110px; "
                        "vertical-align: top; position: relative;"
                    )
                    html += f'<td class="cal-cell" style="{cell_style}" title="Absent">'
                    html += '<div class="cell-body">'
                    html += '<div class="day-top">'
                    html += f'<div class="day-num">{day}</div>'
                    html += f'<span class="status-pill" style="background-color: {pill_color};">{status_labels["absent"]}</span>'
                    html += '</div>'
                    html += '<div class="badge-row">'
                    html += '<span class="hours-pill">0.0h</span>'
                    html += '</div>'
                    html += '<div class="time-range">&nbsp;</div>'
                    html += '<div class="cell-bar">'
                    html += '<div class="hours-track"><span class="hours-fill zero" style="width: 0%;"></span></div>'
                    html += '</div>'
                    html += '</div>'
                    html += '</td>'
                else:
                    # Non-working day (week off)
                    bg_color = colors['weekoff']
                    pill_color = pill_colors['weekoff']
                    cell_style = (
                        f"padding: 8px; border: 1px solid #d7dee7; border-radius: 10px; "
                        f"background-color: {bg_color}; color: #4b5b66; min-height: 110px; "
                        "vertical-align: top; position: relative;"
                    )
                    html += f'<td class="cal-cell" style="{cell_style}">'
                    html += '<div class="cell-body">'
                    html += '<div class="day-top">'
                    html += f'<div class="day-num">{day}</div>'
                    html += f'<span class="status-pill" style="background-color: {pill_color};">{status_labels["weekoff"]}</span>'
                    html += '</div>'
                    html += '<div class="badge-row"></div>'
                    html += '<div class="time-range">&nbsp;</div>'
                    html += '<div class="cell-bar">'
                    html += '<div class="hours-track placeholder"><span class="hours-fill zero" style="width: 0%;"></span></div>'
                    html += '</div>'
                    html += '</div>'
                    html += '</td>'

        html += "</tr>"
    
    html += '</tbody></table>'
    html += '<div class="legend">'
    html += '<div>'
    html += '<div class="legend-title">Status</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="background:{colors["full"]};border-color:#bfe4c9;"></span> Full Day</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="background:{colors["half"]};border-color:#f3dd97;"></span> Half Day</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="background:{colors["short"]};border-color:#f6caa1;"></span> Short Day</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="background:{colors["absent"]};border-color:#f1b5b5;"></span> Absent</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="background:{colors["anomaly"]};border-color:#cbb6f5;"></span> Anomaly</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="background:{colors["weekoff"]};border-color:#dbe1e8;"></span> Week Off</div>'
    html += '</div>'
    html += '<div>'
    html += '<div class="legend-title">Signals</div>'
    html += '<div class="legend-item"><span class="badge badge-late">L</span> Late</div>'
    html += '<div class="legend-item"><span class="badge badge-vlate">VL</span> Very Late</div>'
    html += '<div class="legend-item"><span class="badge badge-early">E</span> Early Departure</div>'
    html += '<div class="legend-item"><span class="badge badge-miss">M</span> Missing Punch Out</div>'
    html += '<div class="legend-item"><span class="badge badge-anom">A</span> Anomaly Flag</div>'
    html += f'<div class="legend-item"><span class="legend-swatch" style="border:2px solid {non_working_border};background:#fff;"></span> Worked on Off-Day</div>'
    html += '<div class="legend-item"><span class="legend-bar"></span> Hours vs Expected</div>'
    html += '</div>'
    html += '</div>'
    html += '</div>'
    
    return html

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="HR Attendance Analytics",
        page_icon="HR",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS - Professional HR Dashboard Styling
    st.markdown("""
        <style>
        .main > div {padding-top: 2rem;}
        
        /* KPI Metric Cards */
        .stMetric {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            transition: box-shadow 0.3s ease;
        }
        .stMetric:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        }
        .stMetric label {
            color: #2E86AB;
            font-weight: 600;
            font-size: 14px;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: #1a1a1a;
            font-size: 32px;
            font-weight: 700;
        }
        .stMetric [data-testid="stMetricDelta"] {
            font-size: 14px;
        }
        
        /* Section Headers */
        h1, h2, h3 {
            color: #2E86AB;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px;
            border-radius: 6px 6px 0 0;
        }
        
        /* DataFrames */
        .dataframe {
            border-radius: 6px;
            overflow: hidden;
        }
        
        /* Captions */
        .stCaption {
            color: #666;
            font-size: 13px;
            font-style: italic;
        }
        </style>
    """, unsafe_allow_html=True)
        
    # Header
    st.title("\U0001F4CA HR Attendance Analytics Dashboard")
    st.markdown("**Complete workforce attendance insights for data-driven HR decisions**")
    st.markdown("---")
    
    # FILE MANAGEMENT & PERSISTENCE
    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001F4CA Data Management")
    uploaded_file = st.sidebar.file_uploader("Update Data Source", type=['xlsx', 'xls'])
    
    data_source = None
    
    # 1. Handle new file upload (Append/Refresh)
    if uploaded_file is not None:
        with st.spinner("Merging and updating data..."):
            merge_ok = DataManager.merge_and_save(uploaded_file, Config.DATA_FILE_PATH)
        if merge_ok:
            st.sidebar.success("\u2705 Data updated successfully!")
            st.cache_data.clear()
            data_source = Config.DATA_FILE_PATH
        else:
            st.stop()
    # 2. Check for existing persistent file
    elif os.path.exists(Config.DATA_FILE_PATH):
        data_source = Config.DATA_FILE_PATH
    
    if data_source is None:
        st.info("Welcome! Please upload an Excel file in the sidebar to initialize the dashboard.")
        st.stop()
    
    # Load data
    with st.spinner("Loading and processing attendance data..."):
        raw_df, daily_df, emp_metrics_df, weekly_overtime_df, monthly_overtime_df = load_and_process_data(data_source)

    if daily_df is None:
        # Error is already displayed in the load function
        st.stop()
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    st.sidebar.header("\U0001F3AF Filters & Controls")
    
    # Date range filter
    min_date = daily_df['Date'].min()
    max_date = daily_df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        daily_filtered = daily_df[
            (daily_df['Date'].dt.date >= start_date) &
            (daily_df['Date'].dt.date <= end_date)
        ]
    else:
        daily_filtered = daily_df
        start_date, end_date = min_date.date(), max_date.date()
    
    # Department filter (multi-select)
    if 'Department' in daily_df.columns:
        dept_options = sorted(daily_filtered['Department'].dropna().unique().tolist())
        selected_depts = st.sidebar.multiselect("Select Department", dept_options)
        
        if selected_depts:
            daily_filtered = daily_filtered[daily_filtered['Department'].isin(selected_depts)]
    
    # Employee filter (multi-select)
    employee_options = sorted(daily_filtered['Employee Full Name'].unique().tolist())
    selected_employees = st.sidebar.multiselect("Select Employee", employee_options)
    
    if selected_employees:
        daily_filtered = daily_filtered[daily_filtered['Employee Full Name'].isin(selected_employees)]
    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001F4CA View Options")
    
    # Toggle options
    exclude_duplicates = st.sidebar.checkbox("Exclude Duplicate Punches", value=True)
    show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", value=False)
    show_late_only = st.sidebar.checkbox("Late Arrivals Only", value=False)
    show_early_only = st.sidebar.checkbox("Early Departures Only", value=False)
    
    # Apply filters
    view_df = daily_filtered.copy()
    
    if show_anomalies_only:
        view_df = view_df[view_df['Has Anomaly']]
    
    if show_late_only:
        view_df = view_df[view_df['Is Late']]
    
    if show_early_only:
        view_df = view_df[view_df['Is Early Departure']]
    # Debug info
    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001F4CA Filter Debug")
    st.sidebar.write(f"Records after filters: {len(view_df)}")
    st.sidebar.write(f"Date range in data: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
    
    # ========================================================================
    # KPI SUMMARY CARDS
    # ========================================================================
    
    st.header("\U0001F4CA Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_employees = view_df['Employee Number'].nunique()
        create_metric_card("\U0001F4CA Total Employees", total_employees)
    
    with col2:
        total_days = count_working_days(start_date, end_date)
        create_metric_card("\U0001F5D3 Working Days", total_days)
    
    with col3:
        avg_hours = view_df['Working Hours'].mean()
        create_metric_card("\u23F1 Avg Daily Hours", f"{avg_hours:.2f}h")
    
    with col4:
        late_pct = (view_df['Is Late'].sum() / len(view_df) * 100) if len(view_df) > 0 else 0
        create_metric_card("\u26A0 Late Arrivals", f"{late_pct:.1f}%")
    
    with col5:
        early_pct = (view_df['Is Early Departure'].sum() / len(view_df) * 100) if len(view_df) > 0 else 0
        create_metric_card("\u26A0 Early Departures", f"{early_pct:.1f}%")
    
    with col6:
        anomaly_count = view_df['Has Anomaly'].sum()
        create_metric_card("\u26A0 Anomalies", anomaly_count)
    
    st.markdown("---")
    
    # ========================================================================
    # TABBED DASHBOARDS
    # ========================================================================
    
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "\U0001F3AF Productivity",
        "\U0001F4C8 Overtime Analysis",
        "\U0001F4C8 Monthly Performance",
        "\U0001F4CA Month-to-Month Comparison",
        "\U0001F5D3 Work Pattern Calendar",
        "\u26A0 Anomalies",
        "\U0001F4CA Data Table",
    ])
    # ------------------------------------------------------------------------
    # TAB 1: PRODUCTIVITY DASHBOARD
    # ------------------------------------------------------------------------
    with tab1:
        st.subheader("\U0001F3AF Productivity Dashboard")
        
        # Recalculate metrics based on filtered data (Date, Employee, Dept)
        # This ensures the Productivity tab respects the Date Range filter
        emp_metrics_filtered = get_productivity_metrics(daily_filtered)
        # Day of week analysis
        dow_summary = get_dow_summary(view_df)
        fig = px.bar(
            x=dow_summary.index,
            y=dow_summary.values,
            title='Average Working Hours by Day of Week',
            labels={'x': 'Day', 'y': 'Avg Hours'},
            color=dow_summary.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shows average working hours across different days of the week")        
        st.plotly_chart(
            plot_employee_ranking(emp_metrics_filtered, 'Total Hours', 10),
            use_container_width=True
        )
        st.caption("Top 10 employees ranked by total working hours")
        
        # Employee performance table
        st.subheader("Employee Performance Summary")

        display_cols = [
            'Employee Full Name', 'Department', 'Total Hours', 'Avg Daily Hours',
            'Total Days', 'Late Count', 'Early Departure Count'
        ]

        if len(emp_metrics_filtered) > 0:
            st.dataframe(
                emp_metrics_filtered[display_cols].sort_values('Total Hours', ascending=False),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No employee data available for the selected filters.")

    # ------------------------------------------------------------------------
    # TAB 2: OVERTIME ANALYSIS
    # ------------------------------------------------------------------------
    with tab2:
        st.subheader("\U0001F4C8 Overtime Analysis")
        st.markdown("**Weekly and monthly overtime hours**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Weekly Overtime")
            
            if not weekly_overtime_df.empty:
                # Filter controls
                years = sorted(weekly_overtime_df['Year'].unique(), reverse=True)
                selected_year_w = st.selectbox("Select Year", years, key="ot_w_year")
                
                available_months_w = sorted(
                    daily_df[daily_df['Date'].dt.year == selected_year_w]['Date'].dt.month.unique(),
                    reverse=True
                )
                
                if not available_months_w:
                    st.info(f"No monthly data available for {selected_year_w}.")
                    selected_week = None
                else:
                    selected_month_w = st.selectbox(
                        "Select Month",
                        available_months_w,
                        format_func=lambda x: calendar.month_name[x],
                        key="ot_w_month"
                    )
                    
                    weeks_in_month = sorted(
                        daily_df[
                            (daily_df['Date'].dt.year == selected_year_w) &
                            (daily_df['Date'].dt.month == selected_month_w)
                        ]['Date'].dt.isocalendar().week.unique(),
                        reverse=True
                    )
                    if not weeks_in_month:
                        st.info(f"No weeks available for {calendar.month_name[selected_month_w]} {selected_year_w}.")
                        selected_week = None
                    else:
                        selected_week = st.selectbox("Select Week Number", weeks_in_month, key="ot_w_num")
                
                # Apply filter
                if selected_week is None:
                    filtered_weekly = weekly_overtime_df.iloc[0:0]
                else:
                    filtered_weekly = weekly_overtime_df[
                        (weekly_overtime_df['Year'] == selected_year_w) & 
                        (weekly_overtime_df['Week'] == selected_week)
                    ]
                
                weekly_chart = plot_overtime_charts(filtered_weekly, 'weekly')
                if weekly_chart:
                    st.plotly_chart(weekly_chart, use_container_width=True)
                else:
                    st.info(f"No overtime recorded for Week {selected_week}, {selected_year_w}.")
            else:
                st.info("No weekly overtime data available.")

        with col2:
            st.markdown("### Monthly Overtime")
            
            if not monthly_overtime_df.empty:
                # Filter controls
                years_m = sorted(monthly_overtime_df['Year'].unique(), reverse=True)
                selected_year_m = st.selectbox("Select Year", years_m, key="ot_m_year")
                
                available_months = sorted(monthly_overtime_df[monthly_overtime_df['Year'] == selected_year_m]['Month'].unique(), reverse=True)
                selected_month = st.selectbox("Select Month", available_months, format_func=lambda x: calendar.month_name[x], key="ot_m_num")
                
                # Apply filter
                filtered_monthly = monthly_overtime_df[
                    (monthly_overtime_df['Year'] == selected_year_m) & 
                    (monthly_overtime_df['Month'] == selected_month)
                ]
                
                monthly_chart = plot_overtime_charts(filtered_monthly, 'monthly')
                if monthly_chart:
                    st.plotly_chart(monthly_chart, use_container_width=True)
                else:
                    st.info(f"No overtime recorded for {calendar.month_name[selected_month]} {selected_year_m}.")
            else:
                st.info("No monthly overtime data available.")

        st.markdown("---")
        st.markdown("### Employee Monthly Overtime Trend")
        if not monthly_overtime_df.empty:
            employee_ot_options = sorted(monthly_overtime_df['Employee Full Name'].unique().tolist())
            selected_emp_ot = st.selectbox("Select Employee", employee_ot_options, key="ot_emp_monthly")
            emp_monthly_ot = monthly_overtime_df[
                monthly_overtime_df['Employee Full Name'] == selected_emp_ot
            ].copy()
            if emp_monthly_ot.empty:
                st.info(f"No monthly overtime data available for {selected_emp_ot}.")
            else:
                emp_monthly_ot['Month Start'] = pd.to_datetime(
                    dict(year=emp_monthly_ot['Year'], month=emp_monthly_ot['Month'], day=1)
                )
                emp_monthly_ot = emp_monthly_ot.sort_values('Month Start')
                fig = px.line(
                    emp_monthly_ot,
                    x='Month Start',
                    y='Monthly Overtime',
                    markers=True,
                    title=f"Monthly Overtime Trend: {selected_emp_ot}",
                    labels={'Month Start': 'Month', 'Monthly Overtime': 'Overtime Hours'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Month-by-month overtime hours for the selected employee")
        else:
            st.info("No monthly overtime data available for trend analysis.")
    # ------------------------------------------------------------------------
    # TAB 3: MONTHLY PERFORMANCE TRACKING
    # ------------------------------------------------------------------------
    with tab3:
        st.subheader("\U0001F4C8 Monthly Performance Tracking")
        st.markdown("**Long-term performance trends and month-over-month analytics**")
        
        # Calculate monthly metrics
        monthly_df = get_monthly_metrics_cached(daily_df)
        
        if len(monthly_df) == 0:
            st.info("No monthly data available for the selected filters.")
        else:
            # Employee selection for trend view
            st.markdown("### Employee Performance Trend")
            employee_options = sorted(daily_df['Employee Full Name'].unique().tolist())
            selected_emp_trend = st.selectbox("Select Employee for Trend Analysis", employee_options, key="trend_emp")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly total hours trend
                fig_hours = plot_monthly_trend(monthly_df, selected_emp_trend, 'Total Hours')
                if fig_hours:
                    st.plotly_chart(fig_hours, use_container_width=True)
                    st.caption("Monthly total working hours trend")
            
            with col2:
                # Monthly attendance days trend
                fig_days = plot_monthly_trend(monthly_df, selected_emp_trend, 'Attendance Days')
                if fig_days:
                    st.plotly_chart(fig_days, use_container_width=True)
                    st.caption("Monthly attendance days count")
            
            st.markdown("---")
            st.markdown("### Monthly Employee Comparison")
            
            # Month selection for comparison
            available_months = sorted(monthly_df['YearMonth'].unique().tolist())
            selected_month = st.selectbox("Select Month for Comparison", available_months, key="comp_month")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Top employees by total hours
                fig_comp_hours = plot_monthly_comparison(monthly_df, selected_month, 'Total Hours', 10)
                if fig_comp_hours:
                    st.plotly_chart(fig_comp_hours, use_container_width=True)
                    st.caption(f"Top 10 employees by total hours - {selected_month}")
            
            with col4:
                # Top employees by average daily hours
                fig_comp_avg = plot_monthly_comparison(monthly_df, selected_month, 'Avg Daily Hours', 10)
                if fig_comp_avg:
                    st.plotly_chart(fig_comp_avg, use_container_width=True)
                    st.caption(f"Top 10 employees by average daily hours - {selected_month}")
            
            # Month-over-month trend indicators
            st.markdown("---")
            st.markdown("### Month-over-Month Change Indicators")
            
            # Calculate MoM changes
            recent_changes = get_recent_changes(monthly_df)
            
            if len(recent_changes) > 0:
                st.dataframe(
                    recent_changes[['Employee Full Name', 'YearMonth', 'Total Hours', 'Prev Total Hours', 
                                  'Hours Change', 'Hours Change %']].head(15),
                    use_container_width=True,
                    height=400
                )
                st.caption(f"Month-over-month changes comparing {available_months[-1]} to previous month")
            else:
                st.info("Insufficient data for month-over-month comparison")
            
            # Monthly summary table
            st.markdown("---")
            st.markdown("### Monthly Performance Summary")
            display_monthly_cols = ['Employee Full Name', 'YearMonth', 'Total Hours', 'Avg Daily Hours', 
                                   'Attendance Days', 'Late Count', 'Early Departure Count']
            st.dataframe(
                monthly_df[display_monthly_cols].sort_values(['YearMonth', 'Total Hours'], ascending=[True, False]),
                use_container_width=True,
                height=400
            )
    
    # ------------------------------------------------------------------------
    # TAB 4: MONTH-TO-MONTH COMPARISON
    # ------------------------------------------------------------------------
    with tab4:
        st.subheader("\U0001F4CA Employee Month-to-Month Comparison")
        st.markdown("**Compare employee performance across two different months**")
        
        # Employee selection
        employee_options = sorted(daily_df['Employee Full Name'].unique().tolist())
        selected_emp_comp = st.selectbox("Select Employee", employee_options, key="comp_emp")
        
        # Calculate monthly metrics if not already done
        if 'monthly_df' not in locals():
            monthly_df = get_monthly_metrics_cached(daily_df)
        
        # Get available months for this employee
        emp_months = sorted(monthly_df[monthly_df['Employee Full Name'] == selected_emp_comp]['YearMonth'].unique().tolist())
        
        if len(emp_months) < 2:
            st.warning(f"Insufficient data for {selected_emp_comp}. Need at least 2 months of data for comparison.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                month1 = st.selectbox("Select First Month", emp_months, key="month1")
            
            with col2:
                # Exclude month1 from month2 options
                month2_options = [m for m in emp_months if m != month1]
                month2 = st.selectbox("Select Second Month", month2_options, key="month2")
            
            # Get data for both months
            month1_data = monthly_df[(monthly_df['Employee Full Name'] == selected_emp_comp) & 
                                    (monthly_df['YearMonth'] == month1)].iloc[0]
            month2_data = monthly_df[(monthly_df['Employee Full Name'] == selected_emp_comp) & 
                                    (monthly_df['YearMonth'] == month2)].iloc[0]
            
            # Calculate changes
            hours_change = month2_data['Total Hours'] - month1_data['Total Hours']
            hours_change_pct = (hours_change / month1_data['Total Hours'] * 100) if month1_data['Total Hours'] > 0 else 0
            
            days_change = month2_data['Attendance Days'] - month1_data['Attendance Days']
            late_change = month2_data['Late Count'] - month1_data['Late Count']
            early_change = month2_data['Early Departure Count'] - month1_data['Early Departure Count']
            
            # Display comparison
            st.markdown("---")
            st.markdown(f"### Comparison: {month1} vs {month2}")
            
            # KPI Cards for comparison
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_hours = f"{hours_change:+.1f}h ({hours_change_pct:+.1f}%)"
                st.metric(
                    "Total Hours",
                    f"{month2_data['Total Hours']:.1f}h",
                    delta=delta_hours,
                    help=f"Month 1: {month1_data['Total Hours']:.1f}h | Month 2: {month2_data['Total Hours']:.1f}h"
                )
            
            with col2:
                st.metric(
                    "Attendance Days",
                    f"{month2_data['Attendance Days']} days",
                    delta=f"{days_change:+d} days",
                    help=f"Month 1: {month1_data['Attendance Days']} days | Month 2: {month2_data['Attendance Days']} days"
                )
            
            with col3:
                st.metric(
                    "Late Arrivals",
                    f"{month2_data['Late Count']}",
                    delta=f"{late_change:+d}",
                    help=f"Month 1: {month1_data['Late Count']} | Month 2: {month2_data['Late Count']}"
                )
            
            with col4:
                st.metric(
                    "Early Departures",
                    f"{month2_data['Early Departure Count']}",
                    delta=f"{early_change:+d}",
                    help=f"Month 1: {month1_data['Early Departure Count']} | Month 2: {month2_data['Early Departure Count']}"
                )
            
            # Visual comparison chart
            st.markdown("---")
            comparison_data = pd.DataFrame({
                'Metric': ['Total Hours', 'Avg Daily Hours', 'Attendance Days', 'Late Count', 'Early Departure Count'],
                month1: [
                    month1_data['Total Hours'],
                    month1_data['Avg Daily Hours'],
                    month1_data['Attendance Days'],
                    month1_data['Late Count'],
                    month1_data['Early Departure Count']
                ],
                month2: [
                    month2_data['Total Hours'],
                    month2_data['Avg Daily Hours'],
                    month2_data['Attendance Days'],
                    month2_data['Late Count'],
                    month2_data['Early Departure Count']
                ]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=month1,
                x=comparison_data['Metric'],
                y=comparison_data[month1],
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                name=month2,
                x=comparison_data['Metric'],
                y=comparison_data[month2],
                marker_color='#2ca02c'
            ))
            
            fig.update_layout(
                title=f'Performance Comparison: {month1} vs {month2}',
                xaxis_title='Metric',
                yaxis_title='Value',
                barmode='group',
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Side-by-side comparison of key metrics across the two selected months")
            
            # Detailed comparison table
            st.markdown("### Detailed Comparison")
            comp_table = pd.DataFrame({
                'Metric': ['Total Hours', 'Average Daily Hours', 'Attendance Days', 
                          'Late Arrivals', 'Early Departures', 'Missing Punch-Outs'],
                month1: [
                    f"{month1_data['Total Hours']:.2f}h",
                    f"{month1_data['Avg Daily Hours']:.2f}h",
                    f"{month1_data['Attendance Days']}",
                    f"{month1_data['Late Count']}",
                    f"{month1_data['Early Departure Count']}",
                    f"{month1_data['Missing Punch Out Count']}"
                ],
                month2: [
                    f"{month2_data['Total Hours']:.2f}h",
                    f"{month2_data['Avg Daily Hours']:.2f}h",
                    f"{month2_data['Attendance Days']}",
                    f"{month2_data['Late Count']}",
                    f"{month2_data['Early Departure Count']}",
                    f"{month2_data['Missing Punch Out Count']}"
                ],
                'Change': [
                    f"{hours_change:+.2f}h ({hours_change_pct:+.1f}%)",
                    f"{(month2_data['Avg Daily Hours'] - month1_data['Avg Daily Hours']):+.2f}h",
                    f"{days_change:+d}",
                    f"{late_change:+d}",
                    f"{early_change:+d}",
                    f"{(month2_data['Missing Punch Out Count'] - month1_data['Missing Punch Out Count']):+d}"
                ]
            })
            st.dataframe(comp_table, use_container_width=True, hide_index=True)
    
    # ------------------------------------------------------------------------
    # TAB 6: ANOMALY DASHBOARD
    # ------------------------------------------------------------------------
    with tab6:
        st.subheader("\u26A0 Anomaly Detection & Analysis")
        
        # Anomaly summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            missing_out = view_df['Missing Punch Out'].sum()
            st.metric("Missing Punch-Outs", missing_out)
        
        with col2:
            short_shifts = view_df['Unusually Short'].sum()
            st.metric("Short Shifts (<4h)", short_shifts)
        
        with col3:
            long_shifts = view_df['Unusually Long'].sum()
            st.metric("Long Shifts (>10h)", long_shifts)
        
        with col4:
            odd_punches = view_df['Odd Punch Count'].sum()
            st.metric("Odd Punch Counts", odd_punches)
        
        # Anomaly details
        st.subheader("Anomaly Records")
        
        anomaly_records = view_df[view_df['Has Anomaly']][
            ['Employee Full Name', 'Date', 'Working Hours', 'Punch Count',
             'Missing Punch Out', 'Unusually Short', 'Unusually Long']
        ].sort_values('Date', ascending=False)
        
        if len(anomaly_records) > 0:
            st.dataframe(anomaly_records, use_container_width=True, height=400)
        else:
            st.success("No anomalies detected in selected period!")
    
    # ------------------------------------------------------------------------
    # TAB 7: DATA TABLE & EXPORT
    # ------------------------------------------------------------------------
    with tab7:
        st.subheader("\U0001F4CA Attendance Data Table")
        
        # Column selector
        available_cols = view_df.columns.tolist()
        default_cols = [
            'Employee Full Name', 'Date', 'First Punch In', 'Last Punch Out',
            'Working Hours', 'Is Late', 'Is Early Departure', 'Shift Type'
        ]
        
        selected_cols = st.multiselect(
            "Select columns to display",
            available_cols,
            default=[col for col in default_cols if col in available_cols]
        )
        
        if selected_cols:
            display_data = view_df[selected_cols].sort_values('Date', ascending=False)
            st.dataframe(display_data, use_container_width=True, height=500)
            
            # Export options
            st.subheader("\U0001F4CA Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_daily = view_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Daily Records (CSV)",
                    data=csv_daily,
                    file_name=f"daily_attendance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Filter employee metrics for current view
                filtered_emp_nums = view_df['Employee Number'].unique()
                emp_export = emp_metrics_df[emp_metrics_df['Employee Number'].isin(filtered_emp_nums)]
                csv_emp = emp_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Employee Summary (CSV)",
                    data=csv_emp,
                    file_name=f"employee_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Monthly summary
                monthly = view_df.groupby(['Employee Full Name', 'Month']).agg({
                    'Working Hours': 'sum',
                    'Date': 'count',
                    'Is Late': 'sum',
                    'Is Early Departure': 'sum'
                }).reset_index()
                monthly.columns = ['Employee', 'Month', 'Total Hours', 'Days Worked', 'Late Count', 'Early Count']
                
                csv_monthly = monthly.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Monthly Summary (CSV)",
                    data=csv_monthly,
                    file_name=f"monthly_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please select at least one column to display")
    
    # ------------------------------------------------------------------------
    # TAB 5: WORK PATTERN CALENDAR
    # ------------------------------------------------------------------------
    with tab5:
        st.subheader("\U0001F5D3 Work Pattern Calendar")
        st.markdown("**Employee-specific work pattern calendar view**")
        
        # Employee selection (respects sidebar filters)
        wp_source_df = daily_filtered.copy()
        employee_options = sorted(wp_source_df['Employee Full Name'].unique().tolist())
        
        if len(employee_options) == 0:
            st.warning("No employees available for the selected filters.")
        else:
            selected_emp_wp = st.selectbox("Select Employee", employee_options, key="wp_cal_emp")
            
            # Date selection
            available_dates = wp_source_df[wp_source_df['Employee Full Name'] == selected_emp_wp]['Date']
            if len(available_dates) > 0:
                min_date = available_dates.min()
                max_date = available_dates.max()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_year_wp = st.selectbox(
                        "Select Year",
                        range(min_date.year, max_date.year + 1),
                        index=len(range(min_date.year, max_date.year + 1)) - 1,
                        key="wp_cal_year"
                    )
                
                with col2:
                    selected_month_wp = st.selectbox(
                        "Select Month",
                        range(1, 13),
                        index=min_date.month - 1 if selected_year_wp == min_date.year else 0,
                        key="wp_cal_month"
                    )

                pattern_message = get_work_pattern_context_text(selected_emp_wp)
                if pattern_message:
                    st.markdown(
                        f"""
                        <div style="background-color: #f3f6fb; border-left: 4px solid #2E86AB;
                                    padding: 10px 14px; border-radius: 6px; color: #1a1a1a;">
                            {pattern_message}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                kpi_data = get_work_pattern_kpis_cached(
                    wp_source_df, selected_emp_wp, selected_year_wp, selected_month_wp
                )
                st.markdown("### Calendar KPIs")
                kpi_row1 = st.columns(3)
                with kpi_row1[0]:
                    create_metric_card("Expected Working Days", int(kpi_data['expected_days']))
                with kpi_row1[1]:
                    create_metric_card(
                        "Actual Worked Days",
                        int(kpi_data['actual_days']),
                        help_text="Counts attendance on expected workdays only."
                    )
                with kpi_row1[2]:
                    create_metric_card("Days Missed", int(kpi_data['missed_days']))

                kpi_row2 = st.columns(3)
                with kpi_row2[0]:
                    create_metric_card("Expected Working Hours", f"{kpi_data['expected_hours']:.1f}h")
                with kpi_row2[1]:
                    create_metric_card(
                        "Actual Worked Hours",
                        f"{kpi_data['actual_hours']:.1f}h",
                        help_text="Totals expected workdays only."
                    )
                with kpi_row2[2]:
                    create_metric_card("Hours Short / Extra", f"{kpi_data['hours_diff']:+.1f}h")

                kpi_row_extra = st.columns(3)
                with kpi_row_extra[0]:
                    create_metric_card(
                        "Total Actual Hours (All Days)",
                        f"{kpi_data['actual_hours_all_days']:.1f}h",
                        help_text="Includes expected and non-working days."
                    )

                kpi_row3 = st.columns(3)
                with kpi_row3[0]:
                    create_metric_card("Late Arrivals", int(kpi_data['late_arrivals']))
                with kpi_row3[1]:
                    create_metric_card("Early Departures", int(kpi_data['early_departures']))
                with kpi_row3[2]:
                    create_metric_card("On-Time Days", int(kpi_data['on_time_days']))

                if kpi_data['worked_non_working_days'] > 0:
                    st.caption(
                        f"Worked on non-working days: {int(kpi_data['worked_non_working_days'])} "
                        "day(s) highlighted in the calendar."
                    )

                st.markdown("---")
                
                # Generate calendar
                calendar_html = get_work_pattern_calendar_cached(
                    wp_source_df,
                    selected_emp_wp,
                    selected_year_wp,
                    selected_month_wp,
                    kpi_data
                )
                
                # Use Streamlit's HTML component for proper rendering (not markdown)
                components.html(calendar_html, height=900, scrolling=False)
                
                st.markdown("---")
                st.markdown("### Work Pattern Distribution")
                
                distribution_df = get_work_pattern_distribution_cached(
                    wp_source_df, selected_emp_wp, selected_year_wp, selected_month_wp
                )
                if len(distribution_df) > 0:
                    color_map = {
                        'Full Day': '#2f9e44',
                        'Half Day': '#caa531',
                        'Short Day': '#cf6d21',
                        'Absent': '#b73b3b',
                        'Week Off': '#7a8794',
                        'Worked on Non-Working Day': '#607d8b'
                    }
                    
                    fig = px.bar(
                        distribution_df,
                        x='Attendance Type',
                        y='Count',
                        title=f'Distribution - {calendar.month_name[selected_month_wp]} {selected_year_wp}',
                        labels={'Count': 'Number of Days', 'Attendance Type': 'Attendance Type'},
                        color='Attendance Type',
                        color_discrete_map=color_map
                    )
                    
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title='Attendance Type',
                        yaxis_title='Number of Days'
                    )
                    
                    fig.update_traces(texttemplate='%{y}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Distribution of attendance types for the selected month (based on work patterns)")
                
                st.info("Legend is embedded in the calendar. OFF tags mark off-day work; L/E/M/A badges highlight timing and punch issues.")
            else:
                st.warning(f"No attendance data found for {selected_emp_wp}")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>HR Attendance Analytics Dashboard v1.0</strong></p>
            <p>Powered by Streamlit | Data processed with Pandas & Plotly</p>
            <p>For support or feedback, contact the HR Analytics Team</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
    
