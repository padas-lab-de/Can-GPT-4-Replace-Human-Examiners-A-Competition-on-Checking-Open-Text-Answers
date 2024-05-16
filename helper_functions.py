import ast
import json
import os
import re
import warnings
from datetime import datetime

import numpy as np
import openai
import pandas as pd
from pandas.errors import PerformanceWarning
from tenacity import (  # , before_sleep_log # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

# Suppress the specific PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)

os.environ['OUTDATED_IGNORE'] = '1'
import pingouin as pg

openai.api_key = os.getenv("OPENAI_API_KEY")



def save_dataframe(df, filename):
    """
    Saves the given DataFrame to both CSV and Excel formats in the current working directory.

    Args:
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The base filename without extension to use for saving the files.

    Returns:
    None
    """
    # Define file paths
    csv_file = f"{filename}.csv"
    excel_file = f"{filename}.xlsx"
    
    # Save as CSV
    df.to_csv(csv_file, index=False)
    # print(f"DataFrame saved as CSV in {csv_file}")
    
    # Save as Excel
    df.to_excel(excel_file, index=False, engine='openpyxl')
    # print(f"DataFrame saved as Excel in {excel_file}")
    
    
def drop_columns_from(df, start_column):
    """
    Drop all columns from the specified start_column to the end of the DataFrame (inclusive).

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.
    start_column (str): The column name from which to start dropping.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns removed.
    """
    # Get the index of the start column
    start_index = df.columns.get_loc(start_column)

    # Get the column names to drop from start_index to the end
    columns_to_drop = df.columns[start_index:]

    # Drop the columns
    df = df.drop(columns=columns_to_drop)
    
    return df

def read_excel_sheets(file_path, sheets=None, return_type='single'):
    """
    Reads specified sheets from an Excel file using pandas.

    :param file_path: str, path to the Excel file.
    :param sheets: str, int, or list, names or indices of the sheets to read.
    :param return_type: str, 'single' to return a single DataFrame (if one sheet is specified),
                        'dict' to return a dictionary of DataFrames (if multiple sheets are specified).
    :return: DataFrame or dict of DataFrames depending on return_type and sheets.
    """
    # Read the sheets based on the provided 'sheets' argument
    try:
        data = pd.read_excel(file_path, sheet_name=sheets)
    except Exception as e:
        print(f"Failed to read the file: {e}")
        return None

    # If multiple sheets are read into a dictionary
    if isinstance(data, dict):
        if return_type == 'single':
            # If user wants a single DataFrame but multiple sheets were requested, raise an error
            raise ValueError("Multiple sheets found but 'single' DataFrame requested. Specify correct 'return_type'.")
        return data
    else:
        if return_type == 'dict':
            # If user expects a dictionary but only one sheet was read, adjust the return structure
            return {sheets: data}
        return data


def extract_points(response):
    # Initialize an empty dictionary to store the points
    points = {}
    
    # Follow #1 when you have the output point 
    # in new line or else follow #2 the second one where I 
    # explicitely mention delimiter
    
    #1
    # Split the response string into individual lines
    # lines = response.split('\n')
    
    #2
    delimiter = '\n' if '\n' in response else '\\n'
    lines = response.split(delimiter)
    
    for line in lines:
        # Extract the sentence number and its corresponding points using regex
        match = re.search(r'Studierendenantwort_(\d+): (\d+)', line)
        # print(match.group(1), match.group(2))
        if match:
            sentence_num = int(match.group(1))
            point = int(match.group(2))

            # Store the points in the dictionary while maintaining the original order
            points[sentence_num] = point
    # print(points)

    # Convert the dictionary to a list of points ordered by sentence number
    ordered_points = [points[i] for i in range(1, len(points) + 1)]

    return ordered_points


'''
Functionality:

Extract the rank from the model's output (which is saved in the csv file)
and create a new column containing (in the same df) the ranks in a list.
'''


def extract_rankings(response):
    # Initialize an empty dictionary to store the rankings
    rankings = {}
    
    # Follow #1  when you have the output rank 
    # in new line or else follow #2 the second one where I 
    # explicitely mention delimiter
    
    #1
    # Split the response string into individual lines
    # lines = response.split('\n')
    
    #2
    delimiter = '\n' if '\n' in response else '\\n'
    lines = response.split(delimiter)
    
    for line in lines:
        # Extract the sentence number and its corresponding rank using regex
        match = re.search(r'Studierendenantwort_(\d+): Rang (\d+)', line)
        if match:
            sentence_num = int(match.group(1))
            rank = int(match.group(2))

            # Store the rank in the dictionary while maintaining the original order
            rankings[sentence_num] = rank


    # Convert the dictionary to a list of rankings ordered by sentence number
    ordered_rankings = [rankings[i] for i in range(1, len(rankings) + 1)]

    return ordered_rankings


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(20))
def chat_with_openai(model_name, messages, temperature, top_p):
    """
    This function initiates a chat with the specified OpenAI model.

    Args:
        model_name (str): The name of the model to be used.
        messages (List[Dict[str, str]]): The list of messages to be sent. Each message is a dictionary with 'role' and 'content'.
        temperature (float): The 'temperature' parameter to be used for the chat completion. Lower values (close to 0) make the output more deterministic, while higher values (close to 1) make it more random.
        top_p (float): The 'top_p' parameter to be used for the chat completion. A float between 0 and 1. 

    Returns:
        dict or str: The chat completion response if successful, otherwise an error message.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            # seed=101,
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )
        # If the response is successful, return the response
        return response

    except openai.error.RateLimitError as e:
        print("Rate limit exceeded. Retrying...")
        raise e  # Retrying is handled by `tenacity` now, so re-raise the exception

    except Exception as e:
        print(f"This Exception: {e}\nRetrying...")
        raise e  # Retrying is handled by `tenacity` now, so re-raise the exception
        

        
def process_dataframe_with_openai_api(df, metadata_df, model_name, temperature, top_p, assessment_prompt_version, assessment_type):

    """
    Processes each row of the given DataFrame by sending prompts to the OpenAI API and stores the response along with 
    additional metadata.

    This function iterates through the DataFrame `df`, extracting prompts from a dynamically determined `input_column`, 
    based on the `assessment_prompt_version` and `assessment_type`, and makes an  API call to OpenAI.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the prompts to be sent to the OpenAI API.
    - metadata_df (pd.DataFrame): A DataFrame to store the metadata of the API interaction.
    - model_name (str): The name of the OpenAI model to be used for generating responses.
    - temperature (float): The temperature parameter for the OpenAI API, controlling randomness.
    - top_p (float): The top_p parameter for the OpenAI API, controlling token selection based on cumulative probability.
    - assessment_prompt_version (str): Version identifier for the prompts, used in naming the columns.
    - assessment_type (str): Type of assessment ('rank' or 'point') to determine the processing method.

    Returns:
    - tuple: A tuple containing two pandas DataFrames:
        - The updated original DataFrame `df` with a new column for model rank/point.
        - The updated `metadata_df` DataFrame containing the metadata for each API call.

    Raises:
    ValueError: if an invalid `assessment_type` is provided or if any of the new columns that
    are to be added to df or metadata_df already exist in these dataframes.

    Exception Handling:
    - In case of an error during the API call or processing, the function prints the error message and sets the 
      corresponding metadata and rankings columns to None or empty values for the affected row.
    """
    
    
    
    # Determine input_column and prefix based on assessment_type
    # input_colum: The name of the column in `df` containing the prompts to be sent to the API.
    # prefix: A string prefix used to name the new columns for the metadata and primary data. This helps in distinguishing 
    #         between different sets of data, especially when processing multiple types of prompts.
    
    if assessment_type == 'rank':
        input_column = f'prompt_{assessment_prompt_version}_rank_assessment'
        prefix = f'prompt_{assessment_prompt_version}_rank_assessment_' # Add dd_mm_yyyy_ if you want in th end. Example: prompt_{assessment_prompt_version}_rank_assessment_22_JUN_2039_
        result_column = f"{prefix}{model_name}_model_ranks"
    elif assessment_type == 'point':
        input_column = f'prompt_{assessment_prompt_version}_point_assessment'
        prefix = f'prompt_{assessment_prompt_version}_point_assessment_' # Add dd_mm_yyyy_ if you want in th end. Example: prompt_{assessment_prompt_version}_rank_assessment_22_JUN_2039_
        result_column = f"{prefix}{model_name}_model_points"
    else:
        raise ValueError("Invalid assessment type. Choose 'rank' or 'point'.")
        
    ''' # it does not make sense to check if the column already exist in the metadata_df as the metadata_df is empty at the first place.   
    # >>> check if overwriting existing column >>>
    # Columns to be added to metadata_df
    metadata_columns = [f"{prefix}{model_name}_{suffix}" for suffix in [
        "complete_input_to_model", "complete_response", "model_output",
        "hyperparameter", "date_and_time", "model_used", "model_ranks" if assessment_type == 'rank' else "model_points"
    ]]

    # Check if any of the new columns already exist in metadata_df or df
    existing_metadata_cols = set(metadata_df.columns).intersection(metadata_columns)
    if existing_metadata_cols or result_column in df.columns:
        raise ValueError(f"Columns {existing_metadata_cols} or Column {result_column} already exist in metadata_df or df.")
    # <<< check if overwriting existing column <<<
    '''
    
    for idx, prompt in enumerate(tqdm(df[input_column], desc=f"Processing {assessment_type} prompt version {assessment_prompt_version}")):
        messages = [{"role": "user", "content": prompt}]
        timestamp = datetime.now()

        try:
            response = chat_with_openai(model_name=model_name, messages=messages, temperature=temperature, top_p=top_p)
            if isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
                model_output = response['choices'][0]['message']['content'] # Only the model output nothing else
                response_str = json.dumps(response)  # Convert complete response to a JSON string
            else:
                model_output = None
                response_str = json.dumps(response)  # Convert response to a JSON string
            
            
            # Dynamically adding metadata to the metadata DataFrame
            metadata_df.loc[idx, f"{prefix}{model_name}_complete_input_to_model"] = messages
            metadata_df.loc[idx, f"{prefix}{model_name}_complete_response"] = response_str
            metadata_df.loc[idx, f"{prefix}{model_name}_model_output"] = model_output
            metadata_df.loc[idx, f"{prefix}{model_name}_hyperparameter"] = json.dumps({"temperature": temperature, "top_p": top_p})
            metadata_df.loc[idx, f"{prefix}{model_name}_date_and_time"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            metadata_df.loc[idx, f"{prefix}{model_name}_model_used"] = model_name
            
            if assessment_type=='rank':
                
                # Convert list of ranks to a string and add to the original DataFrame and metadata
                only_ranks = extract_rankings(model_output) if model_output else []
                
                df.loc[idx, f"{prefix}{model_name}_model_ranks"] = json.dumps(only_ranks) # addding the model rank in the existing df passed in this function
                metadata_df.loc[idx, f"{prefix}{model_name}_model_ranks"] = json.dumps(only_ranks) 
                
            elif assessment_type=='point':
                
                # Convert list of points to a string and add to the original DataFrame and metadata
                only_points = extract_points(model_output) if model_output else []
                
                df.loc[idx, f"{prefix}{model_name}_model_points"] = json.dumps(only_points) # addding the model points in the existing df passed in this function
                metadata_df.loc[idx, f"{prefix}{model_name}_model_points"] = json.dumps(only_points)

        except Exception as e:
            
            
            print(f"Error for prompt {prompt}: {e}")
            
            # Setting the correnspoding columsn to None
            for col in ['complete_input_to_model', 'complete_response', 'model_output', 'hyperparameter', 'date_and_time', 'model_used']:
                metadata_df.loc[idx, f"{prefix}{model_name}_{col}"] = None
            
            if assessment_type == 'rank':
                metadata_df.loc[idx, f"{prefix}{model_name}_model_ranks"] = None
                df.loc[idx, f"{prefix}{model_name}_model_ranks"] = json.dumps([])
            
            elif assessment_type=='point':
                metadata_df.loc[idx, f"{prefix}{model_name}_model_points"]=None
                df.loc[idx, f"{prefix}{model_name}_model_points"]=json.dumps([])


    return df, metadata_df

def extract_and_clean(data):
    """
    Splits the input string on 'gpt', selects the first part, and removes any trailing underscores.

    :param data: str, the input string to process.
    :return: str, the cleaned up string.
    """
    # Step 1: Split the string on 'gpt'
    parts = data.split("gpt")

    # Step 2: Select the first part and remove any trailing underscore
    cleaned_string = parts[0].rstrip("_")

    return cleaned_string


def calculate_statistics_and_save(df, identifier, raters, location_to_save):
    """

    :Function used in point asseessment evaluation


    Calculates statistical metrics and saves the results to a JSON file.

    This function computes Cronbach's alpha and confidence intervals for
    each sublist of raters, and also calculates the correlation matrix
    among the raters. The results, including Cronbach's alpha, confidence
    intervals, and correlation matrix, are stored in a dictionary and then
    saved to a JSON file.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data for calculation.
    - identifier (str): A unique identifier used to name the root key in the
      results dictionary and the output JSON file.
    - raters (list of str): A list of column names in 'df' representing
      different raters.
    - location_to_save (str): Provide the location where you want to save the result along with correlation.

    Returns:
    - dict: A dictionary containing the computed Cronbach's alpha, confidence
      intervals, and correlation matrix.
    - correlation df: A pandas dataframe containing the correlation matrix

    Example usage:
    the_dict = calculate_statistics_and_save(df=my_dataframe,
                                             identifier="my_identifier",
                                             raters=["rater1", "rater2", "rater3"])
    """
    output_lists = create_sublists(raters)
    results_dict = {identifier: {}}

    for lst in output_lists:

        alpha, ci = pg.cronbach_alpha(data=df[lst], ci=0.95)

        # print(f"{lst}\nCronbach's Alpha:{alpha}\nConfidence Interval:{ci}\n")

        key = "_".join(lst)

        results_dict[identifier][key] = {
            "cronbachs_alpha": alpha,
            "confidence_interval": ci.tolist(),
        }
    # if you want correlation also uncomment the below code and return correlation if you want.
    # # Calculate the correlation and return it
    # correlation = df[raters].corr()

    # # Calculate and add the correlation
    # results_dict[identifier]['correlation'] = df[raters].corr().to_dict()

    # Save the results to a JSON file and has correlation also
    save_dict_as_json(d=results_dict, filename=location_to_save)

    return results_dict


def split_list_columns(df, columns_to_split):
    """

    :Function used in point asseessment evaluation

    Split columns containing lists into separate columns for each list element.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_split (list): A list of column names containing lists to be split.
    """
    for column in columns_to_split:
        if column in df.columns:
            # Split the list into separate columns
            max_length = df[column].str.len().max()
            column_names = [f"{column}_studansw_{i+1}" for i in range(max_length)]

            # Expand the lists into new columns and rename them
            df[column_names] = pd.DataFrame(df[column].tolist(), index=df.index)

            # Drop the original column
            # df.drop(column, axis=1, inplace=True)
        else:
            print(f"Column '{column}' not found in DataFrame.")


def convert_string_lists_to_int_lists(df, columns_to_convert):
    """
    
    :Function used in point asseessment evaluation
    
    Convert columns in a DataFrame where each element is a string representation
    of a list into a list of integers.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_convert (list): A list of column names to be converted.
    """
    # Function to check and convert string representation of a list into an actual list
    def convert_string_to_list(item):
        # Check if item is already a list of integers
        if isinstance(item, list) and all(isinstance(x, int) for x in item):
            return item

        # Try converting string to list
        try:
            return ast.literal_eval(item)
        except (ValueError, SyntaxError):
            # Handle cases where the string is not a list representation or other errors
            return []

    # Apply the conversion to each column
    for column in columns_to_convert:
        if column in df.columns:
            if all(isinstance(row, list) and all(isinstance(x, int) for x in row) for row in df[column]):
                print(f"Column '{column}' is already a list of integers.")
            else:
                df[column] = df[column].apply(convert_string_to_list)
                # print(f"Column '{column}' is converted to a list of integers.")
        else:
            print(f"Column '{column}' not found in DataFrame.")


def print_in_box(message: str) -> None:
    """
    Print a given message along with the current directory and timestamp in a box, separated by a horizontal line.

    Parameters:
    message (str): The message to be printed in the box.
    """
    # Get current directory and timestamp
    current_directory = os.getcwd()
    time_now = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

    # Prepare the directory and time information
    dir_info = f"Current directory as of {time_now}:\n{current_directory}"

    # Combine the custom message with the directory information, separated by a line
    combined_message = message + "\n\n" + "-" * len(max(message.split('\n'), key=len)) + "\n" + dir_info

    # Split the combined message into lines
    lines = combined_message.split('\n')
    # Find the length of the longest line
    max_length = max(len(line) for line in lines)
    # Create the top and bottom borders of the box
    top_border = "+" + "-" * (max_length + 2) + "+"
    bottom_border = top_border

    # Print the box with the combined message
    print(top_border)
    for line in lines:
        # Pad each line to the length of the longest line
        padded_line = line + ' ' * (max_length - len(line))
        print("| " + padded_line + " |")
    print(bottom_border)


def save_dict_as_json(d, filename):
    """
    Saves a dictionary as a JSON file, but only if the file does not already exist.

    Parameters:
    d (dict): The dictionary to save.
    filename (str): The path and name of the file to save the dictionary to.

    Raises:
    FileExistsError: If a file with the specified name already exists.
    """

    # Check if the file already exists
    if os.path.exists(filename):
        raise FileExistsError(f"File '{filename}' already exists.")

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the dictionary as a JSON file
    with open(filename, "w") as file:
        json.dump(d, file, indent=4)

    # print_in_box(f"Result saved successfully at\n{filename}")


def save_results_as_nested_json_kendall_w (mean_values, std_deviation_values, root_key, filename):
    """
    
    Used in IRR for ranking in the notebook 'evaluation-rank-assessment.ipynb'
    
    Convert mean and standard deviation values to a nested dictionary and save as JSON.

    :param mean_values: The mean values to be stored (assumed to be in a format convertible to dict).
    :param std_deviation_values: The standard deviation values to be stored (assumed to be in a format convertible to dict).
    :param root_key: The root key for the nested dictionary.
    :param filename: The filename where the JSON should be saved.
    """
    # Convert the values to dictionaries
    mean_values_dict = mean_values.to_dict()
    std_deviation_values_dict = std_deviation_values.to_dict()

    # Create the nested dictionary structure
    nested_results = {
        root_key: {
            "kendall_w_mean_values": mean_values_dict,
            "kendall_w_standard_deviations": std_deviation_values_dict
        }
    }

    save_dict_as_json(d=nested_results,
                      filename=filename)


def read_excel_sheets(file_path, sheets=None, return_type="single"):
    """
    Reads specified sheets from an Excel file using pandas.

    :param file_path: str, path to the Excel file.
    :param sheets: str, int, or list, names or indices of the sheets to read.
    :param return_type: str, 'single' to return a single DataFrame (if one sheet is specified),
                        'dict' to return a dictionary of DataFrames (if multiple sheets are specified).
    :return: DataFrame or dict of DataFrames depending on return_type and sheets.
    """
    # Read the sheets based on the provided 'sheets' argument
    try:
        data = pd.read_excel(file_path, sheet_name=sheets)
    except Exception as e:
        print(f"Failed to read the file: {e}")
        return None

    # If multiple sheets are read into a dictionary
    if isinstance(data, dict):
        if return_type == "single":
            # If user wants a single DataFrame but multiple sheets were requested, raise an error
            raise ValueError(
                "Multiple sheets found but 'single' DataFrame requested. Specify correct 'return_type'."
            )
        return data
    else:
        if return_type == "dict":
            # If user expects a dictionary but only one sheet was read, adjust the return structure
            return {sheets: data}
        return data


def create_sublists(raters):
    """
    Create sublists from a given list, where each sublist contains all elements except one.

    This function iterates through the 'raters' list and for each element, it creates a new list
    that includes all elements from 'raters' except the current one. It returns a list of these sublists.

    Parameters:
    raters (list): A list of elements. The function works with any non-empty list.

    Returns:
    list: A list of sublists, each missing one element from the original 'raters' list.

    Example:
    >>> create_sublists(['A', 'B', 'C', 'D'])
    [['B', 'C', 'D'], ['A', 'C', 'D'], ['A', 'B', 'D'], ['A', 'B', 'C']]
    """

    # if len(raters) != 4:
    # raise ValueError("The input list must contain exactly 4 elements.")

    result_lists = []
    for rater in raters:
        new_list = raters.copy()
        new_list.remove(rater)
        result_lists.append(new_list)

    return result_lists


def calculate_kendall_w_between_n_raters(df, raters, prefix):
    """
    Calculates and adds Kendall's W coefficient as new columns to the provided DataFrame
    for each combination of raters. The function directly modifies the passed DataFrame.

    This function is designed to work with a DataFrame where each rater's ratings are
    provided in separate columns. It calculates the Kendall's W coefficient, a measure of
    agreement between raters, and adds these calculations as new columns to the DataFrame.

    The function updates the original DataFrame in place, adding a new column for each
    combination of raters specified. As a result, there is no need to capture the return
    value of the function since the modifications are applied directly to the passed DataFrame.


    Parameters:
    df (pd.DataFrame): The DataFrame to be modified. Contains ratings from multiple raters.
    raters (list): A list of column names in df, representing different raters.
    prefix (str): A prefix string for naming the new columns.

    Returns:
    pd.DataFrame: The original DataFrame with new columns added, each representing
                  the Kendall's W coefficient for a different combination of raters.

    Raises:
    ValueError: If the number of raters is less than two or if a generated column name
                already exists in the DataFrame.

    Note:
    - The function assumes that the DataFrame and raters list are valid and correctly formatted.
    - Direct modifications to the DataFrame within this function make it efficient for
      scenarios where the original DataFrame needs to be updated across multiple iterations.
    """

    # Check if there are more than one rater
    if len(raters) < 2:
        raise ValueError("At least two raters are required to calculate Kendall's W.")

    # Sort the raters list to ensure consistent column naming
    sorted_raters = sorted(raters)
    raters_identifier = "_".join(sorted_raters)
    new_column_name = prefix + raters_identifier

    # Check if the new column name already exists in the DataFrame
    if new_column_name in df.columns:
        raise ValueError(f"Column '{new_column_name}' already exists in the DataFrame.")

    # Convert the ratings from string to list of integers, if needed
    for rater in raters:
        if not isinstance(df[rater].iloc[0], list):
            df[rater] = df[rater].apply(ast.literal_eval)

    # Define a function to convert a DataFrame row to a suitable format for kendall_w
    def convert_row_to_ratings(df_row, raters):
        ratings = []
        for rater in raters:
            ratings.append(df_row[rater])
        return np.array(ratings)

    # Initialize a list to store Kendall's W for each row
    kendalls_w = []

    # Iterate over each row and calculate Kendall's W
    for index, row in df.iterrows():
        ratings_matrix = convert_row_to_ratings(row, raters)
        w = kendall_w(ratings_matrix)
        kendalls_w.append(w)

    df[new_column_name] = kendalls_w

    # print_in_box(f"kendall_w for {raters} Calculated!")

    return df


# Define the Kendall's W function
def kendall_w(expt_ratings):
    """

    reference: https://stackoverflow.com/a/48916127/12946268

    This function calculates the Kendall's W, which is a measure of agreement between
    multiple rankers. It takes a 2-dimensional numpy array as input, where each row
    corresponds to a rater and each column corresponds to an item that is rated.
    It then calculates the Kendall's W coefficient according to the formula and returns it.

    Parameters:
    expt_ratings (numpy.ndarray): A 2D numpy array where rows are raters and columns are items.

    Returns:
    float: The Kendall's W coefficient indicating the level of agreement.

    Raises:
    ValueError: If the input ratings matrix is not 2-dimensional.

    Example:
    >>> import numpy as np
    >>> ratings = np.array([
            [1, 5, 2, 3, 4],  # rater 1 - 1st row
            [2, 4, 1, 5, 3],  # rater 2 - 2nd row
            [2, 5, 1, 4, 3]   # rater 3 - 3rd row
        ])
    >>> kendall_w(ratings)
    0.8444 (the actual output will vary based on the calculation)

    In this example, ratings are provided by 3 raters (rows) for 5 different items (columns).
    The function calculates and returns the Kendall's W coefficient, which measures the level
    of agreement among the raters' rankings of the items.

    """
    if expt_ratings.ndim != 2:
        raise ValueError("Ratings matrix must be 2-dimensional")
    m = expt_ratings.shape[0]  # raters
    n = expt_ratings.shape[1]  # items rated
    denom = m**2 * (n**3 - n)
    rating_sums = np.sum(expt_ratings, axis=0)
    S = n * np.var(rating_sums)
    return 12 * S / denom
