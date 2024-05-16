import argparse
import builtins
import os

import pandas as pd

from helper_functions import (
    drop_columns_from,
    process_dataframe_with_openai_api,
    read_excel_sheets,
    save_dataframe,
)


def process_and_save_rank_point_data(df, metadata_df, model_name, temperature, top_p, assessment_prompt_version,
                                     assessment_type, data_to_save_path, metadata_to_save_path):
    """
    Processes a DataFrame with the OpenAI API and saves the resulting data and metadata to CSV files.

    Parameters:
    df (pd.DataFrame): The input DataFrame to process.
    metadata_df (pd.DataFrame): The metadata DataFrame.
    model_name (str): The model name for the OpenAI API.
    temperature (float): Temperature parameter for the API.
    top_p (float): Top-p parameter for the API.
    assessment_prompt_version (str): The version of the assessment prompt to use.
    assessment_type (str): The type of assessment to perform ('rank' or 'point').
    data_to_save_path (str): The file path to save the processed data.
    metadata_to_save_path (str): The file path to save the processed metadata.

    Returns:
    None
    """
    # Process the DataFrame with OpenAI API (hypothetical function)
    processed_data, processed_metadata = process_dataframe_with_openai_api(
        df=df,
        metadata_df=metadata_df,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        assessment_prompt_version=assessment_prompt_version,
        assessment_type=assessment_type
    )
    

    folder_name = "newly_generated_data"

    # Check if the folder already exists
    # Needed to save the processed data and metadata
    if not os.path.exists(folder_name):
        # Create the folder
        os.mkdir(folder_name)
        print(f"The folder '{folder_name}' was created.")
    else:
        pass
    
    save_dataframe(df=processed_data,
                   filename=data_to_save_path)
    
    save_dataframe(df=processed_metadata,
                   filename=metadata_to_save_path)



def main():
    
    parser = argparse.ArgumentParser(description='Process and save rank and point data.')
    
    parser.add_argument("-d", "--data",
                        help="Read the complete prompt from V1 to V12 from -> Data file path default is (default: %(default)s)",
                        default="./original_data/Data_complete_Can_GPT_Replace_Human_Examiners.xlsx",
                        required=False)
    parser.add_argument("-m", "--model_name",
                        help="Model name for the OpenAI API (default: %(default)s)",
                        default="gpt-4-0125-preview",
                        required=False)
    
    parser.add_argument("-t", "--temperature",
                        help="Temperature parameter for the API (default: %(default)s)",
                        default=0,
                        required=False) 
    parser.add_argument("-tp", "--top_p",
                        help="Top-p parameter for the API (default: %(default)s)",
                        default=1,
                        required=False)
    parser.add_argument("--data_to_save_path", 
                        help="The directory path to save the generated data (default: %(default)s)",
                        default='./newly_generated_data/data_gpt4_ranks_points',
                        required=False)
    
    parser.add_argument("--metadata_to_save_path", 
                        help="The directory path to save the metadata (default: %(default)s)",
                        default='./newly_generated_data/metadata_gpt4_ranks_points',
                        required=False)

    parser.add_argument("-q", "--quiet",
                        help="If this flag used: Suppress all output from the script. Use this option if you prefer a silent run without any console messages.",
                        action="store_true")
    


    
    args = parser.parse_args()
    
    if args.quiet:
        # Override the print function
        builtins.print = lambda *args, **kwargs: None
    
    data = read_excel_sheets(file_path=args.data,
                             sheets='Robustness & Extensions') # Load or create your DataFrame here
    # >>> post processing the data >>> #
    # Drop columns from the DataFrame here and keep only the relevant columns
    # i.e. no columns with ranks or points should be there from GPT.
    complete_rank_point_data = drop_columns_from(data, 'prompt_v1_rank_assessment_gpt4-ranks-run1')
    # complete_rank_point_data = complete_rank_point_data[:2].copy() # for testing 
    # <<< post processing the data <<< #
    complete_rank_point_metadata = pd.DataFrame()  # Load or create your metadata DataFrame here
    
    type_of_assessments = ['rank', 'point']
    prompt_versions = [1, 2, 3, 4, 5, 6.1, 6.2, 7, 8, 9, 10, 11, 12]
    
    # Check if the file path to save the processed data or metadata already exists. 
    # If it does, raise an error.
    if os.path.exists(f"{args.data_to_save_path}.csv") or os.path.exists(f"{args.metadata_to_save_path}.csv"):
        raise FileExistsError("The file path to save the processed data or metadata already exists.")
    
    for type_of_assessment in type_of_assessments:
        for prompt_version in prompt_versions:
            process_and_save_rank_point_data(
                df=complete_rank_point_data,
                metadata_df=complete_rank_point_metadata,
                model_name=args.model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                assessment_prompt_version=f"v{prompt_version}",
                assessment_type=type_of_assessment,
                data_to_save_path=args.data_to_save_path,
                metadata_to_save_path=args.metadata_to_save_path
            )


# Example usage
if __name__ == "__main__":
    exit(main())

