import pandas as pd
import numpy as np
import ollama
from sklearn.impute import KNNImputer, SimpleImputer
#from fancyimpute import IterativeImputer  # MICE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import chi2_contingency
from fuzzywuzzy import fuzz
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.stats import shapiro, ttest_ind, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import chardet
import io
persona = 'DA'; #default choice
explanation_log = [] #BO
stats_log =[] #DA
domainText = []
dom = " "
"""
file_path = "" #"/Users/zehraahmed/Downloads/untitled folder/Automobile Sales.csv"
# Detect encoding
with open(file_path, "rb") as f:
    raw_data = f.read(10000)
    detected_encoding = chardet.detect(raw_data)['encoding']
print(f"Detected encoding: {detected_encoding} for {file_path}")
        
        # Read dataset
if file_path.endswith(".csv"):
    try:
        df = pd.read_csv(file_path, encoding=detected_encoding, low_memory=False)
    except (UnicodeDecodeError, ValueError):
        print("⚠️ Encoding issue detected. Retrying with ISO-8859-1.")
        df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)  # Fallback encoding
                
    excel_path = file_path.replace(".csv", ".xlsx")
    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"CSV converted to Excel: {excel_path}")
elif file_path.endswith((".xls", ".xlsx")):
    df = pd.read_excel(file_path, engine="openpyxl")

# Drop empty columns
df = df.dropna(axis=1, how='all')
print(df.shape)
"""
def select_persona(p):
    persona = p
    if p == 'BO':
        explanation_log.append("Business Owner: Simple Actionable Insights Generated")
    else:
        stats_log.append("Data Analyst: Statistical + Analytical Insights Generated")
    #print(p)
    return persona

def detect_data_domain(df):
    """Detects the domain of the dataset and provides a response."""
    try:
        # Check if dataframe is empty
        if df.empty:
            return "❌ The dataset is empty. Please upload a valid dataset."

        # Extract columns and sample data for prompt
        columns = df.columns.tolist()
        sample_data = df.head(3).to_string()

        prompt = f"""
        You are NexBI, an intelligent BI consultant. Your task is to identify the domain of the dataset and provide a confident, professional, and engaging response.

        ### Task:
        1. Identify the domain of the dataset based on column names.
        2. Mention key columns that helped you determine the domain.
        3. Provide a brief explanation (1-2 sentences) of what this dataset is used for.
        4. End with a natural follow-up question to guide the user forward.

        ### Format:
        1. Concise and to the point.
        2. Avoid unnecessary details.
        3. Focus on essential columns and insights.
        4. Make the response sound human, approachable, and professional.
        5. End with a question that invites further exploration.

        ### Examples:
        #### Input:
        Columns: ['Order ID', 'Customer Name', 'Product Category', 'Sales Amount']
        #### Output:
        "This is a sales dataset. The columns ‘Order ID,’ ‘Customer Name,’ and ‘Sales Amount’ suggest it tracks customer purchases and sales performance. Would you like to explore sales trends or customer demographics?"

        #### Input:
        Columns: ['Transaction ID', 'Account Balance', 'Expense Category']
        #### Output:
        "This appears to be a financial dataset. The columns ‘Transaction ID,’ ‘Account Balance,’ and ‘Expense Category’ suggest it's tracking financial transactions. Would you like me to check for spending patterns?"

        #### Input:
        Columns: ['Patient ID', 'Diagnosis Code', 'Treatment Plan']
        #### Output:
        "This is a healthcare dataset. Columns like ‘Patient ID,’ ‘Diagnosis Code,’ and ‘Treatment Plan’ suggest it tracks patient medical information. Should I look for missing data or analyze treatment outcomes?"

        ### Now, analyze the following dataset:
        Column Names: {columns}
        Sample Data: {sample_data}
        """

        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        domain = response["message"]["content"].strip()

        if not domain:
            return "❌ Unable to detect the domain of the dataset. Please check the dataset for sufficient information."
        #explanation_log.append("Detecting Data Domain..." + domain + "\n")
        #stats_log.append("Detecting Data Domain..." + domain + "\n")
        #domainText.append("Detecting Data Domain..." + domain + "\n")
        dom = domain
        return domain
    
    except Exception as e:
        return f"❌ Error while detecting domain: {str(e)}"
    
def clean_numeric_columns(df, numeric_threshold=0.05):
    """
    Cleans columns by removing rows with non-numeric values in columns that have 
    a majority of numeric values and converts them to numeric data types.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_threshold (float): The threshold percentage to consider a column 
                                   as numeric (default is 5%).
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with rows removed where numeric columns
                      contained non-numeric values.
    """
    dropped_rows_details = []  # List to store details of invalid rows (column, value, index)

    for col in df.columns:
        # Check if the column is numeric or can be considered numeric
        non_numeric_count = df[col].apply(pd.to_numeric, errors='coerce').isna().sum()
        total_count = len(df)

        # If the proportion of non-numeric values is below the threshold, treat as numeric
        if non_numeric_count / total_count < numeric_threshold:
            # Find rows with non-numeric values (excluding NaN)
            non_numeric_rows = df[col][df[col].apply(pd.to_numeric, errors='coerce').isna() & df[col].notna()]

            # Log the rows causing the issue
            for index, value in non_numeric_rows.items():
                dropped_rows_details.append({
                    "column": col,
                    "index": index,
                    "value": value
                })

            # Remove the rows where non-numeric values are found, but leave NaN rows intact
            df = df.dropna(subset=[col], how='any')  # Remove rows where NaN exists in the column
            
            # Alternatively, you can drop rows with non-numeric values directly (if needed)
            df = df[~df[col].apply(pd.to_numeric, errors='coerce').isna()]
            explanation_log.append("Checking for Numeric Columns containing symbols or text\n" + str(dropped_rows_details) + "\n")
            stats_log.append("Checking for Numeric Columns containing symbols or text\n" + str(dropped_rows_details) + "\n")


    return df

def standardize_categories(df, stats_log, explanation_log):
    """
    Standardizes text columns by using the mode value for inconsistent entries and logs the changes.
    """
    for col in df.select_dtypes(include=['object']):
        # Find the most common value (mode)
        mode_value = df[col].mode()[0]
        
        # Count how many rows don't match the mode
        non_matching_rows = df[col].apply(lambda x: fuzz.ratio(str(x).strip().lower(), str(mode_value).strip().lower()) <= 85).sum()
        
        # Log the number of non-matching rows and what mode value was chosen
        if non_matching_rows > 0:
            stats_log.append(f"Standardized {non_matching_rows} rows in column '{col}' to mode value '{mode_value}'.")
            explanation_log.append(f"Rows with inconsistent values in '{col}' were converted to mode value '{mode_value}'.")
        
        # Apply fuzzy matching to standardize data close to the mode
        df[col] = df[col].apply(lambda x: mode_value if pd.isnull(x) or fuzz.ratio(str(x).strip().lower(), str(mode_value).strip().lower()) > 85 else x)
        
    return df, stats_log, explanation_log

def correct_currency_symbols(df, stats_log, explanation_log):
    """
    Corrects inconsistent currency symbols (e.g., $, €, £) in the DataFrame columns by standardizing them.
    Args:
        df (pd.DataFrame): The input DataFrame.
        stats_log (list): List to store insights.
        explanation_log (list): List to store explanations of changes.
    Returns:
        pd.DataFrame: DataFrame with corrected currency symbols.
        list: Updated stats_log with insights.
        list: Updated explanation_log with details of changes made.
    """
    for col in df.select_dtypes(include=['object']):  # Only process string columns
        # Ensure the column is treated as string data (convert to string if necessary)
        column_data = df[col].astype(str)
        
        # Check if the column contains currency symbols
        if column_data.str.contains(r'[$€£,]', regex=True, na=False).any():  # Handle NaN values gracefully
            # Log the problematic rows
            problematic_rows = column_data[column_data.str.contains(r'[$€£,]', regex=True, na=False)].to_list()
            stats_log.append(f"Found {len(problematic_rows)} rows with inconsistent currency symbols in column '{col}'.")
            explanation_log.append(f"Rows with inconsistent currency symbols detected in column '{col}': {problematic_rows}")
            
            # Remove currency symbols (standardize or clean as needed)
            df[col] = df[col].replace({r'[$€£,]': ''}, regex=True)
            
            # Optionally, convert the cleaned column to numeric values if needed
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log the changes made
            stats_log.append(f"Removed currency symbols in column '{col}'.")
            explanation_log.append(f"Currency symbols removed from column '{col}' and converted to numeric values.")

    return df, stats_log, explanation_log

def correct_units(df, stats_log, explanation_log):
    """
    Convert units to a common format (e.g., lbs to kg) based on mode value, and logs the changes.
    """
    for col in df.select_dtypes(include=['object']):
        # Skip columns that shouldn't be processed (e.g., non-numeric text columns)
        if df[col].str.contains(r'kg|lbs|g|oz', regex=True, na=False).any():
            # Get the mode value, ensuring it's a string
            mode_value = str(df[col].mode()[0]) if not pd.isna(df[col].mode()[0]) else ''
            
            # Count how many rows need unit conversion
            rows_with_units = df[col].str.contains(r'lbs|kg|g|oz', regex=True, na=False).sum()
            
            if rows_with_units > 0:
                stats_log.append(f"Converted {rows_with_units} rows from inconsistent units to mode value '{mode_value}' in column '{col}'.")
                explanation_log.append(f"Rows with inconsistent units in column '{col}' were converted to mode value '{mode_value}'.")
            
            # Convert units based on mode value (e.g., converting lbs to kg)
            if 'lbs' in mode_value:
                # Convert 'lbs' to 'kg'
                df[col] = df[col].replace(r'(\d+)\s*lbs', lambda m: str(float(m.group(1)) * 0.453592) + ' kg', regex=True)
            elif 'kg' in mode_value:
                # Optionally, handle other unit conversions like 'g' to 'kg', etc.
                df[col] = df[col].replace(r'(\d+)\s*g', lambda m: str(float(m.group(1)) / 1000) + ' kg', regex=True)
            
            # Remove units after conversion and convert to numeric
            df[col] = df[col].replace(r'kg|lbs|g|oz', '', regex=True)
            
            # Convert to numeric, and keep the original values for non-convertible entries
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, stats_log, explanation_log


def correct_dates(df, stats_log, explanation_log):
    """
    Corrects inconsistent or incomplete dates in a DataFrame column, and logs the changes.
    - Handles partial dates like '1/10/' by inferring or adding the missing parts.
    - Converts all valid dates to a standard format.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        stats_log (list): List to store insights.
        explanation_log (list): List to store explanations of changes.
        
    Returns:
        pd.DataFrame: DataFrame with corrected date columns.
        list: Updated stats_log with insights.
        list: Updated explanation_log with details of changes made.
    """
    
    for col in df.select_dtypes(include=['object']):
        # Check if the column contains date-like strings before trying conversion
        if df[col].str.match(r'\d{1,2}/\d{1,2}/\d{2,4}', na=False).any():  # Match common date formats (e.g., 1/10/2025)
            try:
                # Try converting the column to datetime, forcing errors to NaT
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
            except Exception as e:
                # If there's an error in converting, skip this column
                continue
        
            # After conversion, check for missing or incomplete date values
            incomplete_dates = df[col].isna()
            
            if incomplete_dates.any():
                # Log the rows that have incomplete or invalid dates
                invalid_date_count = incomplete_dates.sum()
                stats_log.append(f"Found {invalid_date_count} incomplete or invalid dates in column '{col}'.")
                explanation_log.append(f"{invalid_date_count} invalid or incomplete dates were detected in '{col}'.")
                
                # Log the rows with incomplete date information for the user's attention
                problematic_dates = df[col][incomplete_dates].to_list()
                stats_log.append(f"Dates needing attention in column '{col}': {problematic_dates}.")
                explanation_log.append(f"Please review the following dates in column '{col}' that require user intervention: {problematic_dates}.")
                
    # Returning the dataframe without any automatic fixes, allowing user intervention
    return df, stats_log, explanation_log


def clean_text_data(df, stats_log, explanation_log):
    """
    Standardizes text data using the majority rule and fuzzy matching, and logs the changes.
    """
    for col in df.select_dtypes(include=['object']):
        # Find the mode of the column
        mode_value = df[col].mode()[0]
        
        # Count how many rows don't match the mode
        non_matching_rows = df[col].apply(lambda x: fuzz.ratio(str(x).strip().lower(), str(mode_value).strip().lower()) <= 85).sum()
        
        # Log the number of non-matching rows and what mode value was chosen
        if non_matching_rows > 0:
            stats_log.append(f"Standardized {non_matching_rows} rows in column '{col}' to mode value '{mode_value}'.")
            explanation_log.append(f"Rows with inconsistent values in '{col}' were converted to mode value '{mode_value}'.")
        
        # Apply fuzzy matching to standardize data close to the mode
        df[col] = df[col].apply(lambda x: mode_value if pd.isnull(x) or fuzz.ratio(str(x).strip().lower(), str(mode_value).strip().lower()) > 85 else x)
        
    return df, stats_log, explanation_log

def handle_inconsistent_data(df, stats_log, explanation_log):
    """
    Applies multiple data cleaning functions to handle inconsistent data by majority rule.
    Logs the changes in stats_log and explanation_log.
    """
    df= clean_numeric_columns(df, 0.05)

    # Standardize text categories (cleaning spelling issues and whitespace)
    df, stats_log, explanation_log = standardize_categories(df, stats_log, explanation_log)
   
    # Correct currency symbols and clean numeric columns
    #df, stats_log, explanation_log = correct_currency_symbols(df, stats_log, explanation_log)
    
    """
    df, stats_log, explanation_log = correct_units(df, stats_log, explanation_log)
    df.to_excel('cleaned_dataset5.xlsx', index=False)
    # Correct date formats
    df, stats_log, explanation_log = correct_dates(df, stats_log, explanation_log)
    df.to_excel('cleaned_dataset6.xlsx', index=False)
    """
    # Clean text data by majority rule and fuzzy matching
    df, stats_log, explanation_log = clean_text_data(df, stats_log, explanation_log)
    
    return df, stats_log, explanation_log

def handle_missing_values_DA_BO(df, explanation_log, stats_log):
    def split_df_by_dtype(df):
        """Splits the DataFrame into numeric and categorical (object) columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(exclude=[np.number]).columns
        return df[numeric_cols], df[object_cols]

    def logistic_regression_mar(df, column):
        print(column)
        """Checks if missing values in a numeric column are Missing At Random (MAR) 
        using Logistic Regression. A high model accuracy suggests MAR."""
        if df[column].isnull().sum() == 0:
            print(df[column])
            return None
        
        df_numeric = df.select_dtypes(include=[np.number]).copy()
        df_numeric["missing_indicator"] = df[column].isna().astype(int)
        df_numeric.drop(columns=[column], inplace=True)
        df_numeric.dropna(inplace=True)
        
        if df_numeric.shape[1] < 2:
            return None  # Not enough data for regression

        X = df_numeric.drop(columns=["missing_indicator"])
        y = df_numeric["missing_indicator"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        return 1 if model.score(X_scaled, y) > 0.6 else 0

    def chi_square_mar_test(df, column):
        """Checks if missing values in a categorical column are MAR 
        using a Chi-Square test. A significant p-value (< 0.05) suggests MAR."""
        if df[column].isnull().sum() == 0:
            return None
        
        df["missing_indicator"] = df[column].isna().astype(int)
        contingency_table = pd.crosstab(df["missing_indicator"], df[column], dropna=False)
        
        if contingency_table.shape[1] < 2:
            return None  # Not enough categories for statistical test
        
        _, p_value, _, _ = chi2_contingency(contingency_table)
        return 1 if p_value < 0.05 else 0

    def little_mcar_test(df):
        """Performs Little's MCAR test to check if missing values are 
        completely random (MCAR). A p-value > 0.05 suggests MCAR."""
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.isnull().sum().sum() == 0:
            return None  # No missing values in numeric data

        imputer = SimpleImputer(strategy="mean")
        df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
        diff = df_numeric - df_imputed
        squared_diff = diff.pow(2).sum().sum()
        df_degrees = (df_numeric.shape[0] - 1) * (df_numeric.shape[1] - 1)
        
        # Ensure the contingency table is valid before applying the test
        contingency_table = np.array([[squared_diff, df_degrees]])
        if np.any(contingency_table == 0):  
            return None  # Avoid division errors in statistical tests

        p_value = 1 - chi2_contingency(contingency_table)[1]
        
        return 1 if p_value > 0.05 else 0

    def chi_square_mcar_test(df, column):
        """Uses a Chi-Square test to check if missing values in a categorical 
        column are completely random (MCAR). A p-value > 0.05 suggests MCAR."""
        if df[column].isnull().sum() == 0:
            return None
        
        missing_mask = df[column].isna().astype(int)
        contingency_table = pd.crosstab(missing_mask, df[column], dropna=False)
        
        if contingency_table.shape[1] < 2:
            return None  # Not enough categories to conduct a valid test
        
        _, p_value, _, _ = chi2_contingency(contingency_table)
        return 1 if p_value > 0.05 else 0

    def classify_missingness(df):
        """Creates a structured table that labels each column as MAR, MCAR, or MNAR."""
        numeric_df, object_df = split_df_by_dtype(df)
        results = []
        
        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue
            
            mar = logistic_regression_mar(df, column) if column in numeric_df.columns else chi_square_mar_test(df, column)
            mcar = little_mcar_test(df) if column in numeric_df.columns else chi_square_mcar_test(df, column)
            
            # If neither MAR nor MCAR, classify as MNAR (Missing Not At Random)
            mnar = 1 if (mar == 0 and mcar == 0) else 0
            results.append([column, mar or 0, mcar or 0, mnar])
        
        return pd.DataFrame(results, columns=["Column", "MAR", "MCAR", "MNAR"])

    # Generate the missingness classification table
    missingness_table = classify_missingness(df)
    df_cleaned = df.copy()
    missingness_table = missingness_table.set_index('Column')  # Set 'Column' as index for easy access
    explanation_log.append("Checking for missing values and fixing them... ")
    stats_log.append("Checking for missing values and fixing them...")

    for column in missingness_table.index:
        missing_type = missingness_table.loc[column]
        if missing_type['MCAR'] == 1:
            # MCAR: Use mean for numeric, mode for categorical
            if df_cleaned[column].dtype == 'O':  
                imputer = SimpleImputer(strategy='most_frequent')  # Mode for categorical
                df_cleaned[[column]] = imputer.fit_transform(df_cleaned[[column]])
                explanation_log.append(f"Column '{column}' had missing values completely at random (MCAR). We filled them using the most frequent value (mode).")
                stats_log.append(f"MCAR Imputation: Mode (most frequent value) used for categorical column '{column}'.")
            else:  
                imputer = SimpleImputer(strategy='mean')  
                df_cleaned[[column]] = imputer.fit_transform(df_cleaned[[column]])
                explanation_log.append(f"Column '{column}' had missing values completely at random (MCAR). We filled them using the mean.")
                stats_log.append(f"MCAR Imputation: Mean imputation applied to numeric column '{column}'.")

        elif missing_type['MAR'] == 1:
            # MAR: Use regression for numeric, mode for categorical
            #print(f"Processing column: {column}")

            if df_cleaned[column].dtype == 'O':
                # Categorical column - use the most frequent value (mode)
                imputer = SimpleImputer(strategy='most_frequent')  
                df_cleaned[[column]] = imputer.fit_transform(df_cleaned[[column]])
                explanation_log.append(f"Column '{column}' had missing values dependent on other variables (MAR). We filled them using the most frequent value (mode).")
                stats_log.append(f"MAR Imputation: Mode (most frequent value) used for categorical column '{column}'.")
             #   print(f"Column '{column}' is categorical. Mode imputation completed.")
            else:
                # Numeric column - use regression-based imputation
              #  print(f"Column '{column}' is numeric. Checking predictors for regression imputation...")

                # Prepare predictors (all numeric columns except the target column)
                predictors = df_cleaned.drop(columns=[column]).select_dtypes(include=[np.number])
                target = df_cleaned[column]

                # Impute missing values in predictors (you can choose strategy like 'mean' or 'median')
                predictor_imputer = SimpleImputer(strategy='mean')  # or 'median'
                predictors_imputed = predictor_imputer.fit_transform(predictors)

                # Debugging: Check if there are missing values in predictors
               # print(f"Missing values in predictors after imputation:\n{pd.DataFrame(predictors_imputed).isnull().sum()}")

                # Now check if predictors have no missing values
                if pd.DataFrame(predictors_imputed).isnull().sum().sum() == 0:
                #    print(f"No missing values in predictors for column '{column}'.")

                    # Check if there are enough non-null values in the target column to fit the model
                    known_values = target.notnull()
                 #   print(f"Known values in target column '{column}': {known_values.sum()} / {len(known_values)}")

                    if known_values.sum() > 0:
                        model = LinearRegression()
                        model.fit(predictors_imputed[known_values], target[known_values])

                        # Predict the missing values in the target column
                        df_cleaned.loc[~known_values, column] = model.predict(predictors_imputed[~known_values])

                        # Check the updated column for missing values
                  #      print(f"Missing values after regression imputation in column '{column}': {df_cleaned[column].isnull().sum()}")

                        explanation_log.append(f"Column '{column}' had missing values dependent on other variables (MAR). We used regression-based imputation to predict missing values.")
                        stats_log.append(f"MAR Imputation: Linear regression imputation applied to numeric column '{column}'.")
                        print(f"Regression imputation completed for column '{column}'.")
                    else:
                        print(f"Not enough known values to fit regression model for column '{column}'.")
                else:
                    print(f"Missing values detected in predictors after imputation for column '{column}', skipping regression.")



        elif missing_type['MNAR'] == 1:
            # MNAR: Flag for categorical, KNN for numeric
            if df_cleaned[column].dtype == 'O':
                df_cleaned[column] = df_cleaned[column].fillna('Missing')
                explanation_log.append(f"Column '{column}' had missing values that were not random (MNAR). We flagged them as a separate category 'Missing'.")
                stats_log.append(f"MNAR Handling: New category 'Missing' introduced for categorical column '{column}'.")
            else:
                knn_imputer = KNeighborsRegressor(n_neighbors=5)
                predictors = df_cleaned.drop(columns=[column]).select_dtypes(include=[np.number])
                target = df_cleaned[column]
                
                if predictors.isnull().sum().sum() == 0:
                    known_values = target.notnull()
                    knn_imputer.fit(predictors[known_values], target[known_values])
                    df_cleaned.loc[~known_values, column] = knn_imputer.predict(predictors[~known_values])
                    explanation_log.append(f"Column '{column}' had missing values that were not random (MNAR). We used KNN imputation to estimate them.")
                    stats_log.append(f"MNAR Imputation: KNN imputation applied to numeric column '{column}'.")
    if 'missing_indicator' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop('missing_indicator', axis=1)

    return df_cleaned, explanation_log, stats_log


def drop_extreme_outliers(df, explanation_logs, stats_logs):
    # Create a copy of the dataframe to avoid modifying the original one
    stats_logs.append("Checking for Outliers...")
    explanation_logs.append("Checking for Extreme Values that might affect analysis...")
    df_cleaned = df.copy()

    # Identify numeric columns
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

    # Loop over each numeric column
    for col in numeric_cols:
        # Calculate the Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        
        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for extreme outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify the rows that are outliers
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]

        # Log the number of outliers detected
        num_outliers = outliers.shape[0]
        
        if num_outliers > 0:
            # Remove the rows that contain outliers
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            
            # Explanation log for why the rows were removed
            stats_log.append(f"Column '{col}' had {num_outliers} extreme outliers based on IQR. Rows with values outside the bounds ({lower_bound}, {upper_bound}) were dropped.")
            
            # Stats log for exactly how many rows were dropped
            explanation_logs.append(f"Column '{col}' had {num_outliers} extreme values. This would have impacted distribution, hence all {num_outliers} rows were dropped.")

    # Return the cleaned dataframe with extreme outliers dropped
    return df_cleaned



def perform_eda_and_stat_tests(df, explanation_logs, stats_logs):
    # Data type handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    stats_logs.append("Starting Exploratory Data Analysis and Statistical Tests to determine trends, patterns, and insights...")
    explanation_logs.append("Exploring data to determine trends, patterns, areas for improvement and insights...")
    
    # Univariate Analysis (Numeric Columns)
    for col in numeric_cols:
        # Calculate basic statistics
        mean = df[col].mean()
        median = df[col].median()
        mode = df[col].mode()[0] if not df[col].mode().empty else "N/A"
        std_dev = df[col].std()
        skewness = df[col].skew()
        kurtosis = df[col].kurt()
        print("pass 1")
        # Conditional business interpretation
        # Mean vs Median Insight
        if mean < median:
            mean_vs_median = "The average is lower than the median, suggesting a few unusually low values are pulling the overall average down."
        elif mean > median:
            mean_vs_median = "The average is higher than the median, likely due to a few high values inflating the trend."
        else:
            mean_vs_median = "The average and median are closely aligned, indicating a well-balanced distribution."

        # Skewness Insight
        if skewness < -1:
            skew_insight = "The distribution is heavily skewed to the left, meaning most values are high, but some low outliers are dragging down the average."
        elif -1 <= skewness < -0.5:
            skew_insight = "There’s moderate left skewness, with a few lower values slightly affecting the overall trend."
        elif -0.5 <= skewness <= 0.5:
            skew_insight = "The data appears fairly symmetrical with no significant skew."
        elif 0.5 < skewness <= 1:
            skew_insight = "There’s moderate right skewness, indicating a few higher values are influencing the data."
        else:
            skew_insight = "The distribution is heavily skewed to the right, where most values are low but a few high outliers may be distorting insights."

        # Kurtosis Insight
        if kurtosis < 3:
            kurtosis_insight = "The data has lighter tails than normal, meaning fewer extreme values and a more predictable pattern."
        elif kurtosis == 3:
            kurtosis_insight = "The distribution has normal kurtosis, which suggests a standard level of variability and peaks."
        else:
            kurtosis_insight = "The data has heavier tails, indicating the presence of outliers or extreme values—something to monitor closely."

        # Standard Deviation interpretation
        variability = "low variation and consistency" if std_dev < 1 else "noticeable variation that may need further investigation"

        # Append professional explanation
        explanation_logs.append(
            f"""Column '{col}': The average (Mean) is {mean:.2f}. {mean_vs_median}. The median is {median: .2f}, serving as the midpoint of the data. The most common value (Mode) is {mode}, helping identify frequent behavior or repeated outcomes. The standard deviation is {std_dev:.2f}, indicating {variability}. Skewness is {skewness:.2f}. {skew_insight} Kurtosis is {kurtosis:.2f}. {kurtosis_insight}
            """
        )
        stats_logs.append(
            f"""Performing Univariate Analysis for column '{col}': The average (Mean) is {mean:.2f}. {mean_vs_median}. The median is {median: .2f}, serving as the midpoint of the data. The most common value (Mode) is {mode}, helping identify frequent behavior or repeated outcomes. The standard deviation is {std_dev:.2f}, indicating {variability}. Skewness is {skewness:.2f}. {skew_insight} Kurtosis is {kurtosis:.2f}. {kurtosis_insight}
            """
        )

        print("pass 2")
        # Insights for skewness
        if abs(skewness) > 1:
            explanation_logs.append(f"Column '{col}' has significant skewness. This might indicate a non-normal distribution. Consider transforming it before analysis.")
            stats_logs.append(f"Column '{col}' has significant skewness, suggesting a non-normal distribution. Consider transforming it before analysis.")

        # Normality test (Shapiro-Wilk Test)
        if df[col].dropna().shape[0] < 5000:
            
            stat, p_value = shapiro(df[col].dropna())

            # Log raw result
            stats_logs.append(f"Shapiro-Wilk Test for normality in column '{col}' — p-value: {p_value:.4f}")

            # Interpret results
            if p_value < 0.05:
                explanation_logs.append(
                    f"The Shapiro-Wilk test returned a p-value of {p_value:.4f}, suggesting the data **does not follow a normal distribution**. "
                    "This may imply the presence of outliers, a skewed pattern, or irregular distribution—something to consider when choosing forecasting models or evaluating averages."
                )
                stats_logs.append(f"Column '{col}' does not follow a normal distribution (p-value: {p_value:.4f}).")
                explanation_logs.append("\n")
                stats_logs.append("\n")
            else:
                explanation_logs.append(
                    f"The Shapiro-Wilk test returned a p-value of {p_value:.4f}, indicating the data is **approximately normally distributed**. "
                    "This suggests that common statistical methods and averages are reliable and representative for this column."
                )
                stats_logs.append(f"Column '{col}' follows a normal distribution (p-value: {p_value:.4f}).")
                explanation_logs.append("\n")
                stats_logs.append("\n")
    print("pass 3")
    explanation_logs.append("Now exploring categorical columns individually:\n")
    stats_logs.append("Performing Univariate Analysis for Categorical Columns\n")
    for col in categorical_cols:
        # Mode and imbalance check
        mode = df[col].mode()[0]
        value_counts = df[col].value_counts()
        imbalance = value_counts.min() / value_counts.max() if value_counts.max() > 0 else 0

        # Technical stats log
        stats_logs.append(f"Column '{col}' — Mode: {mode}, Imbalance Ratio: {imbalance:.2f}")

        # Business-friendly insight
        explanation_logs.append(
            f"Column '{col}': The most frequent category is '{mode}', which appears most often in the dataset. "
        )

        # Imbalance ratio interpretation
        if imbalance < 0.2:
            imbalance_insight = (
                "There is a strong imbalance between categories — meaning one category heavily dominates the rest. It could affect analysis. Consider grouping rare categories or applying resampling techniques."
                "This could lead to biased outcomes or overlooked segments in analysis or decision-making."
            )
        elif 0.2 <= imbalance < 0.5:
            imbalance_insight = (
                "There is a moderate imbalance among categories. While one group is more frequent, others still hold relevance."
            )
        elif 0.5 <= imbalance < 0.8:
            imbalance_insight = (
                "There’s a mild imbalance, with fairly even distribution across categories — a good sign for representativeness."
            )
        else:
            imbalance_insight = (
                "The categories are well-balanced, meaning insights are likely to be evenly representative across all segments."
            )

        explanation_logs.append(imbalance_insight)
        stats_logs.append(imbalance_insight)
    print("pass 4")
    explanation_logs.append("\nNow exploring the relationshop between two or more categorical columns:\n")
    stats_logs.append("\nPerforming Bivariate / Multivariate Analysis (Correlation & Pairwise Analysis) for Categorical Columns\n")
    # Bivariate / Multivariate Analysis (Correlation & Pairwise Analysis)
    correlation_matrix = df[numeric_cols].corr()

    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                correlation = correlation_matrix[col1][col2]
                stats_logs.append(f"Correlation between '{col1}' and '{col2}': {correlation:.2f}")

                # High absolute correlation
                if abs(correlation) > 0.8:
                    if correlation > 0:
                        explanation_logs.append(
                            f"Columns '{col1}' and '{col2}' exhibit a **strong positive correlation** (correlation = {correlation:.2f}). "
                            "This suggests they move in the same direction — when one increases, the other tends to increase as well. "
                            "It may be **redundant** to include both in dashboards or reports, as they reflect similar underlying behavior."
                        )
                        stats_logs.append(
                            f"Columns '{col1}' and '{col2}' exhibit a **strong positive correlation** (correlation = {correlation:.2f}). "
                            "This suggests they move in the same direction — when one increases, the other tends to increase as well. "
                            "It may be **redundant** to include both in dashboards or reports, as they reflect similar underlying behavior."
                        )
                    else:
                        explanation_logs.append(
                            f"Columns '{col1}' and '{col2}' show a **strong negative correlation** (correlation = {correlation:.2f}). "
                            "This indicates an inverse relationship — as one goes up, the other tends to go down. "
                            "This insight could be useful for **diagnosing trade-offs** or identifying balancing effects in the business."
                        )
                        stats_logs.append(
                            f"Columns '{col1}' and '{col2}' show a **strong negative correlation** (correlation = {correlation:.2f}). "
                            "This indicates an inverse relationship — as one goes up, the other tends to go down. "
                            "This insight could be useful for **diagnosing trade-offs** or identifying balancing effects in the business."
                        )
                elif 0.4 < abs(correlation) <= 0.8:
                    if correlation > 0:
                        explanation_logs.append(
                            f"Columns '{col1}' and '{col2}' have a **moderate positive correlation** (correlation = {correlation:.2f}). "
                            "There’s a noticeable relationship, but they may still represent distinct business patterns worth monitoring separately."
                        )
                        stats_logs.append(
                            f"Columns '{col1}' and '{col2}' have a **moderate positive correlation** (correlation = {correlation:.2f}). "
                            "There’s a noticeable relationship, but they may still represent distinct business patterns worth monitoring separately."
                        )
                    else:
                        explanation_logs.append(
                            f"Columns '{col1}' and '{col2}' have a **moderate negative correlation** (correlation = {correlation:.2f}). "
                            "They may influence each other inversely to some extent, which could be helpful in understanding **balancing factors** within operations."
                        )
                        stats_logs.append(
                            f"Columns '{col1}' and '{col2}' have a **moderate negative correlation** (correlation = {correlation:.2f}). "
                            "They may influence each other inversely to some extent, which could be helpful in understanding **balancing factors** within operations."
                        )
                elif 0.2 < abs(correlation) <= 0.4:
                    explanation_logs.append(
                        f"Columns '{col1}' and '{col2}' show a **weak correlation** (correlation = {correlation:.2f}). "
                        "There may be a minor relationship, but it's not strong enough to draw confident conclusions."
                    )
                    stats_logs.append(
                        f"Columns '{col1}' and '{col2}' show a **weak correlation** (correlation = {correlation:.2f}). "
                        "There may be a minor relationship, but it's not strong enough to draw confident conclusions."
                    )
                else:
                    explanation_logs.append(
                        f"Columns '{col1}' and '{col2}' show **little to no correlation** (correlation = {correlation:.2f}). "
                        "These variables likely operate independently of one another in the current dataset."
                    )
                    stats_log.append(
                        f"Columns '{col1}' and '{col2}' show **little to no correlation** (correlation = {correlation:.2f}). "
                        "These variables likely operate independently of one another in the current dataset."
                    )

    print("pass 5")
     # Perform T-tests for numeric columns, grouped by categorical columns
    for col in numeric_cols:
        for cat_col in categorical_cols:
            if df[cat_col].nunique() == 2:  # Ensure the categorical column has only 2 unique categories
                group1 = df[df[cat_col] == df[cat_col].unique()[0]][col].dropna()
                group2 = df[df[cat_col] == df[cat_col].unique()[1]][col].dropna()
                
                if len(group1) > 1 and len(group2) > 1:  # Ensure there is enough data in each group for a valid test
                    stat, p_value = ttest_ind(group1, group2)
                    #explanation_logs.append(f"T-Test between '{col}' and categories of '{cat_col}' - p-value: {p_value}")
                    #stats_logs.append(f"T-Test between '{col}' and categories of '{cat_col}' - p-value: {p_value}")
                    
                    if p_value < 0.05:
                        explanation_logs.append(f"There is a significant difference between the categories of '{cat_col}' in '{col}' (p-value: {p_value: .4f}).")
                        stats_logs.append(f"Significant difference detected between the categories of '{cat_col}' in '{col}' (p-value: {p_value: .4f}).")
                        
                        # Actionable insights based on significance
                        explanation_logs.append(f"Visualize the relationship between '{col}' and '{cat_col}' to better understand the difference.")
                        stats_logs.append(f"For deeper analysis, plot '{col}' against '{cat_col}' using a box plot or bar chart.")
                        
                        # Suggest filtering or drilling down for more insight
                        explanation_logs.append(f"Consider filtering data by '{cat_col}' to explore the specific differences in '{col}' between the two categories.")
                        stats_logs.append(f"Consider slicing the data by '{cat_col}' (e.g., Category 1 vs. Category 2) to drill down into the sales behavior between groups.")

                        # Business insights based on the analysis
                        explanation_logs.append(f"Since there is a significant difference, you may need to tailor business strategies specifically for these categories. For example, marketing strategies or resource allocation might need to be adjusted based on the group differences.")
                        stats_logs.append(f"From a business perspective, this significant difference in '{col}' suggests that different strategies might be needed for each category of '{cat_col}'. Consider segmenting strategies based on this insight.")
            
                
    
    print("pass 6")
    # Statistical Tests (ANOVA - for comparing more than 2 groups)
    # Set a threshold for unique values (e.g., skip categorical columns with more than 100 unique values)
    unique_threshold = 100  # You can adjust this threshold based on your data

    # Statistical Tests (ANOVA - for comparing more than 2 groups)
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        for cat_col in categorical_cols:
            # Skip columns with too many unique categories
            if len(df[cat_col].unique()) > unique_threshold:
                print(f"Skipping '{cat_col}' due to too many unique categories.")
                continue
            
            for num_col in numeric_cols:
                # Perform ANOVA between categorical and numeric columns (to compare means across multiple categories)
                grouped_data = [df[df[cat_col] == category][num_col].dropna() for category in df[cat_col].unique()]
                stat, p_value = f_oneway(*grouped_data)
                
                if p_value < 0.05:
                    explanation_logs.append(f"There is a significant difference in '{num_col}' based on '{cat_col}' (p-value: {p_value: .4f}).")
                    stats_logs.append(f"Significant difference detected in '{num_col}' based on '{cat_col}' (p-value: {p_value: .4f}).")
                    
                    # Actionable insights based on significance
                    explanation_logs.append(f"Visualize the relationship between '{num_col}' and '{cat_col}' to better understand the difference.")
                    stats_logs.append(f"To better understand the differences, consider visualizing '{num_col}' vs. '{cat_col}' using a box plot, bar chart, or violin plot.")
                    
                    # Suggest filtering or drilling down for more insight
                    explanation_logs.append(f"Consider filtering data by '{cat_col}' to explore the differences in '{num_col}' between categories.")
                    stats_logs.append(f"Consider slicing the data by '{cat_col}' (e.g., comparing Category 1, Category 2, etc.) to drill down into the numeric behavior between groups.")
                    
                    # Business insights based on the analysis
                    explanation_logs.append(f"Since there is a significant difference, you may need to tailor your business strategies for different categories of '{cat_col}'.")
                    stats_logs.append(f"From a business perspective, the significant difference in '{num_col}' across categories of '{cat_col}' suggests that targeted strategies may be necessary based on group differences.")

                    # Perform Tukey's HSD Test
                    try:
                        tukey = pairwise_tukeyhsd(endog=df[num_col].dropna(), groups=df[cat_col].dropna(), alpha=0.05)
                        tukey_summary = tukey.summary()
                        
                        # Interpret significant differences
                        significant_pairs = [
                            (row[0], row[1], row[5])  # group1, group2, reject
                            for row in tukey._results_table.data[1:]
                            if row[5] == True
                        ]
                        
                        if significant_pairs:
                            differences = "; ".join([f"'{g1}' vs '{g2}'" for g1, g2, _ in significant_pairs])
                            explanation_logs.append(
                                f"**Statistically significant differences** were found in '{num_col}' across the following pairs of '{cat_col}' categories: {differences}. "
                                "This implies that these groups behave differently and may require different business approaches, pricing, or attention."
                            )
                            stats_logs.append(
                                f"**Statistically significant differences** were found in '{num_col}' across the following pairs of '{cat_col}' categories: {differences}. "
                                "This implies that these groups behave differently and may require different business approaches, pricing, or attention."
                            )
                        else:
                            explanation_logs.append(
                                f"No statistically significant differences were found in average '{num_col}' across '{cat_col}' categories. "
                                "This suggests the groups perform similarly with respect to this metric."
                            )
                            stats_logs.append(
                                f"No statistically significant differences were found in average '{num_col}' across '{cat_col}' categories. "
                                "This suggests the groups perform similarly with respect to this metric."
                            )
                    except Exception as e:
                        """explanation_logs.append(
                            f"Could not perform Tukey's HSD test for '{cat_col}' vs '{num_col}' — possibly due to insufficient or uneven data. ({str(e)})"
                        )
                        stats_logs.append(
                            f"Could not perform Tukey's HSD test for '{cat_col}' vs '{num_col}' — possibly due to insufficient or uneven data. ({str(e)})"
                        ) """

    print("pass 7")
    # Chi-Square Test (Categorical Variables)
    if len(categorical_cols) > 1:
        # Set a percentage threshold (e.g., skip columns with more than 20% unique values)
        percentage_threshold = 0.2  # 20% of unique values, you can adjust this as needed

        # Loop over categorical columns
        for cat_col1 in categorical_cols:
            # Calculate the percentage of unique values in the column
            unique_percentage1 = len(df[cat_col1].unique()) / len(df)

            # Skip the column if the percentage of unique values exceeds the threshold
            if unique_percentage1 > percentage_threshold:
                continue
            
            for cat_col2 in categorical_cols:
                # Skip if the column has more unique values than the threshold
                unique_percentage2 = len(df[cat_col2].unique()) / len(df)
                if unique_percentage2 > percentage_threshold:
                    continue

                if cat_col1 != cat_col2:
                    # Chi-square test between two categorical columns
                    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
                    stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    if p_value < 0.05:
                        explanation_logs.append(f"There is a significant association between '{cat_col1}' and '{cat_col2}' (p-value: {p_value: .4f}).")
                        stats_logs.append(f"Significant association found between '{cat_col1}' and '{cat_col2}' (p-value: {p_value: .4f}).")
                        
                        # Actionable insights based on significance
                        explanation_logs.append(f"Consider exploring the relationship between '{cat_col1}' and '{cat_col2}' for better targeting in business strategies.")
                        stats_logs.append(f"To explore the association, consider visualizing the relationship between '{cat_col1}' and '{cat_col2}' using a stacked bar chart or mosaic plot.")
                        
                        # Business insights based on the analysis
                        explanation_logs.append(f"Since a significant association exists, you may want to explore joint categories for devising strategies.")
                        stats_logs.append(f"From a business perspective, the significant association between '{cat_col1}' and '{cat_col2}' may suggest the need to combine or segment certain categories for more effective targeting.")

                            


    # Summary logs
    explanation_logs.append("EDA and statistical tests completed. Check logs for detailed insights.")
    stats_logs.append("EDA and statistical tests completed. Review the logs for detailed results and interpretations.")

    return df, explanation_logs, stats_logs

from fpdf import FPDF

# Define a function to call Mistral 7B via Ollama and get insights
def generate_bi_report(df, log, domain):
    columns = df.columns.tolist()
    dataset = df.head(3).to_string()
    
    # Define the prompts for Business Owner and Data Analyst personas
    prompt = f"""
        Act as an experienced Business Intelligence Consultant analyzing a dataset from the {domain} domain. Here is a small sample of the dataset, {dataset} along with column names {columns} You are presenting findings to key stakeholders, ensuring insights are tailored, relevant, and directly tied to their business objectives. Your responses should be structured as a professional, engaging conversation—avoid sounding scripted or generic. Instead, provide actionable, data-driven recommendations and discuss potential business impacts.

        The raw insights are:
        {log}

        Rewrite these insights to:

        Sound like a human, expert consultant delivering them in a direct stakeholder conversation.
        Contextualize each insight based on the dataset's domain and typical business concerns in this industry.
        Highlight key trends, anomalies, and their possible business implications.
        Offer practical recommendations in a structured yet engaging manner.
        Use a confident, consultative tone—avoid robotic or overly technical phrasing unless needed for clarity.
        Your response should make the insights sound natural, persuasive, and highly relevant to business decision-makers.
        
        """
    

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    report = response["message"]["content"].strip()
    
    return report


# Define a function to convert the report into a PDF file
import re
from fpdf import FPDF

from fpdf import FPDF
import re
class CustomPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_color = (255, 245, 225)

    def header(self):
        """ Applies the background color to every page. """
        self.set_fill_color(*self.bg_color)
        self.rect(0, 0, 210, 297, 'F')

def save_report_to_pdf(report, filename="BI_Report.pdf"):
    """
    Converts the generated BI report to a well-formatted, downloadable PDF.
    
    :param report: The formatted BI report.
    :param filename: The name of the PDF file to save.
    :return: None
    """
    # Remove unsupported characters (non-ASCII)
    sanitized_report = re.sub(r'[^\x00-\x7F]+', '', report)

    pdf = CustomPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(255, 245, 225)
    pdf.rect(0, 0, 210, 297, 'F')

    # Set title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Datalyze", ln=True, align='C')
    pdf.ln(10)  # Line break after title

    # Set font for content
    pdf.set_font("Arial", size=10)

    # Ensure each log entry starts on a new line and wraps properly
    for line in sanitized_report.split("\n"):
        pdf.multi_cell(0, 7, line)  # Adjust cell height for better spacing
        pdf.ln(1)  # Small line break between entries

    # Save PDF
    pdf.output(filename)






"""

dom = detect_data_domain(df) #uncomment
#print("\n".join(explanation_log))
#print("\n".join(stats_log))
#df, _, _ = handle_inconsistent_data(df, stats_log, explanation_log)
#df.to_excel('cleaned_dataset4.xlsx', index=False)
df, _, _ = handle_missing_values_DA_BO(df, explanation_log, stats_log)
df_cleaned = drop_extreme_outliers(df, explanation_log, stats_log)
df, _, _ = perform_eda_and_stat_tests(df_cleaned, explanation_log, stats_log)
#print("\n".join(explanation_log))
#print("\n".join(stats_log))


# Generate report for Business Owner or Data Analyst
persona = "BO"  # You can change this to "data_analyst" if needed

if persona=='DA':
    #report = generate_bi_report(stats_log, domainText)
    domainText.append("\n" + str(stats_log))
    #domainText.append(str(report) + "\n") 
    my_string = ' '.join(map(str, domainText))
else:
    report = generate_bi_report(df, explanation_log, dom)
    #my_string = dom + "\n" + report
    #print(my_string)
    #domainText.append("\n" + str(explanation_log))
    #domainText.append(str(report) + "\n") 
    #my_string = ' '.join(map(str, domainText))
    #my_string = "\n".join(explanation_log)
    my_string = f"{dom}\n\n{report}\n\n" + "\n".join(explanation_log)
# Save the generated report to a PDF
save_report_to_pdf(my_string, filename="Business_Intelligence_ReportNew6.pdf")

print("Report generated and saved as PDF.")
"""