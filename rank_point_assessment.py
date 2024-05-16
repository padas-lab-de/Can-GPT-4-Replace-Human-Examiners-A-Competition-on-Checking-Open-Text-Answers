import argparse
import builtins
import time

import pandas as pd
from tqdm import tqdm

from helper_functions import (  # print_in_box,
    calculate_kendall_w_between_n_raters,
    calculate_statistics_and_save,
    convert_string_lists_to_int_lists,
    create_sublists,
    extract_and_clean,
    read_excel_sheets,
    save_results_as_nested_json_kendall_w,
    split_list_columns,
)


def rank_assessment(data, per_question_irr=False, newly_generated_data_gpt4=False):
    
    # All the columns in the data file that have the model ranks only
    if not newly_generated_data_gpt4:
        # original_data
        model_ranks_columns = ['prompt_v1_rank_assessment_gpt4-ranks-run1', 'prompt_v1_rank_assessment_gpt4-ranks-run2',
                                'prompt_v2_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v3_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v4_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v5_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v6.1_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v6.2_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v7_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v8_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v9_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v10_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v11_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v12_rank_assessment_gpt-4-0125-preview_model_ranks']
    else:
        # newly_generated_data
        model_ranks_columns =['prompt_v1_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v2_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v3_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v4_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v5_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v6.1_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v6.2_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v7_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v8_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v9_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v10_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v11_rank_assessment_gpt-4-0125-preview_model_ranks',
                                'prompt_v12_rank_assessment_gpt-4-0125-preview_model_ranks']
    
    

    # Create a tqdm object with an initial description
    progress_bar = tqdm(model_ranks_columns, desc="Starting processing")

    for gpt_4_ranks in progress_bar:
        
        progress_bar.set_description(f"Processing {extract_and_clean(gpt_4_ranks)}")
        
        time.sleep(1)  # Simulate time delay of processing


        working_df_rank = data[['Questions', 'Set-Number',
                           gpt_4_ranks,
                           'human-examiner-1-ranks', 'human-examiner-2-ranks', 'human-examiner-3-ranks']].copy()
        
        raters = ['human-examiner-1-ranks','human-examiner-2-ranks', 'human-examiner-3-ranks',  gpt_4_ranks]
        
        output_lists = create_sublists(raters)


        for lst in output_lists:    

            calculate_kendall_w_between_n_raters(df=working_df_rank,
                                                raters=lst,
                                                prefix='Kendall_w_')
        
        included_columns = [working_df_rank.columns[-4],
                            working_df_rank.columns[-3],
                            working_df_rank.columns[-2],
                            working_df_rank.columns[-1]]
        
        identifier_to_save_the_result = extract_and_clean(gpt_4_ranks)
        
        mean_values = working_df_rank[included_columns].mean()
        std_deviation_values = working_df_rank[included_columns].std()

        if not newly_generated_data_gpt4:
            if gpt_4_ranks == "prompt_v1_rank_assessment_gpt4-ranks-run2":
                identifier_to_save_the_result = identifier_to_save_the_result + '_run2'
            elif gpt_4_ranks == "prompt_v1_rank_assessment_gpt4-ranks-run1":
                identifier_to_save_the_result = identifier_to_save_the_result + '_run1'
        else: 
            pass

        if per_question_irr == False:  # just the main results
            
            # Choose the base directory based on the boolean value
            base_dir = "newly_generated_data_results" if newly_generated_data_gpt4 else "original_data_results"
            
            save_results_as_nested_json_kendall_w(mean_values=mean_values,
                                                      std_deviation_values=std_deviation_values,
                                                      root_key=f"{identifier_to_save_the_result}_questions1to6",
                                                      filename=f"./{base_dir}/rank-assessment/pooled/{identifier_to_save_the_result}/{identifier_to_save_the_result}_questions1to6.json")
                
                
            
        elif per_question_irr == True: # per question IRR and main results
                
            
            for question_counter in range(1, 7):
                
                mean_values = working_df_rank[working_df_rank['Questions'] == f'Question{question_counter}'][included_columns].mean()

                std_deviation_values = working_df_rank[working_df_rank['Questions'] == f'Question{question_counter}'][included_columns].std()
                
                identifier_to_save_the_result_per_questions = f"{identifier_to_save_the_result}_question{question_counter}"
                
                base_dir = "newly_generated_data_results" if newly_generated_data_gpt4 else "original_data_results"    
                
                save_results_as_nested_json_kendall_w(mean_values=mean_values,
                                            std_deviation_values=std_deviation_values,
                                            root_key=identifier_to_save_the_result_per_questions,
                                            filename=f"./{base_dir}/rank-assessment/per-question/{identifier_to_save_the_result}/{identifier_to_save_the_result_per_questions}.json")
                
                
            
            
        




def point_assessment(data, per_question_irr=False, newly_generated_data_gpt4=False):
    
    def _melt_dataframe(df, id_vars, value_vars, var_name, value_name):
        """
        Melts the given dataframe on specified variables.

        :param df: DataFrame to be melted
        :param id_vars: List of columns to use as identifier variables. These columns will be left unchanged.
                        These columns are typically the identifiers for each row.
        :param value_vars: List of columns to unpivot. Columns that you want to melt down into rows.
                        By specifying certain columns, you indicate which columns should be unpivoted.
        :param var_name: Name to use for the ‘variable’ column.
                        Rename the new column created by melt that stores the 'column headers' i.e  ['Studierendenantwort_1',  'Studierendenantwort_2', 
                        'Studierendenantwort_3',  'Studierendenantwort_4',  'Studierendenantwort_5'] which were melted down.
        :param value_name: Name to use for the ‘value’ column.
                        Renames the new column created by melt that contains the values from the melted columns
        :return: Melted DataFrame
        """
        return pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )

    def _process_dataframe(df, columns):

        convert_string_lists_to_int_lists(df=df, columns_to_convert=columns)
        split_list_columns(df=df, columns_to_split=columns)

        return df
    
    # All the columns in the data file that have the model ranks only
    if not newly_generated_data_gpt4:

        # original_data
        model_points_columns = ['prompt_v1_point_assessment_gpt4-points-run1', 'prompt_v1_point_assessment_gpt4-points-run2',
                                'prompt_v2_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v3_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v4_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v5_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v6.1_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v6.2_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v7_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v8_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v9_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v10_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v11_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v12_point_assessment_gpt-4-0125-preview_model_points']
        
    else:
        # newly_generated_data
        model_points_columns = ['prompt_v1_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v2_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v3_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v4_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v5_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v6.1_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v6.2_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v7_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v8_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v9_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v10_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v11_point_assessment_gpt-4-0125-preview_model_points',
                                'prompt_v12_point_assessment_gpt-4-0125-preview_model_points']
    
    progress_bar = tqdm(model_points_columns, desc="Starting processing")
    
    for gpt_4_points in progress_bar:

        progress_bar.set_description(f"Processing {extract_and_clean(gpt_4_points)}")

        time.sleep(1)  
    
        working_df_points = data[['Questions', 'Set-Number',
                                  'Student_Answer_1', 'Student_Answer_2', 'Student_Answer_3', 'Student_Answer_4', 'Student_Answer_5',
                                  gpt_4_points, # column that has the MODEL'S point
                                  'human-examiner-1-points', 'human-examiner-2-points', 'human-examiner-3-points']].copy()
        # print(working_df_points.head(6))
        
        columns_in_question = [
            gpt_4_points,
            'human-examiner-1-points', 'human-examiner-2-points', 'human-examiner-3-points'
        ]
        
        working_df_points_processed = _process_dataframe(df=working_df_points ,
                                                        columns=columns_in_question) 
        
                
        # Define value variables for each category
        student_answers = [f'Student_Answer_{i}' for i in range(1, 6)]
        model_points = [f'{gpt_4_points}_studansw_{i}' for i in range(1, 6)]
        human1_points = [f'human-examiner-1-points_studansw_{i}' for i in range(1, 6)]
        human2_points = [f'human-examiner-2-points_studansw_{i}' for i in range(1, 6)]
        human3_points = [f'human-examiner-3-points_studansw_{i}' for i in range(1, 6)]


        df_student_answer = _melt_dataframe(df=working_df_points_processed,
                                        id_vars=['Questions', 'Set-Number'], # These are columns in the dataframe that you want to keep as is. These columns are usually identifiers for each row.
                                        value_vars=student_answers, # Columns that you want to unpivot or melt down into rows.
                                        var_name='Student_Answer_Number',
                                        value_name='Student_Answer_Text')


        df_model_points = _melt_dataframe(df=working_df_points_processed,
                                        id_vars=['Questions', 'Set-Number'], 
                                        value_vars=model_points, 
                                        var_name=f"melted_{gpt_4_points}_studansw_n",
                                        value_name=f"melted_{gpt_4_points}")

        df_human1_point = _melt_dataframe(df=working_df_points_processed,
                                    id_vars=['Questions', 'Set-Number'], 
                                    value_vars=human1_points, 
                                    var_name='human-examiner-1-points_studansw_n_points', 
                                    value_name='human1_points')

        df_human2_point = _melt_dataframe(df=working_df_points_processed,
                                        id_vars=['Questions', 'Set-Number'], 
                                        value_vars=human2_points, 
                                        var_name='human-examiner-2-points_studansw_n_points', 
                                        value_name='human2_points')

        df_human3_point = _melt_dataframe(df=working_df_points_processed,
                                        id_vars=['Questions', 'Set-Number'], 
                                        value_vars=human3_points, 
                                        var_name='human-examiner-3-points_studansw_n_points', 
                                        value_name='human3_points')


        # Concatenating all the melted dataframes into one final dataframe
        working_df_points_processed_ready_for_cronbachs_alpha = pd.concat([df_student_answer[['Questions', 'Set-Number', 'Student_Answer_Text']],
                            df_model_points[f"melted_{gpt_4_points}"], #According to what is in df_model_points value_name
                            df_human1_point['human1_points'],
                            df_human2_point['human2_points'],
                            df_human3_point['human3_points']], 
                            axis=1)
        
        # return working_df_points_processed_ready_for_cronbachs_alpha
        identifier_to_save_the_result = extract_and_clean(gpt_4_points)
        
        if not newly_generated_data_gpt4:        
            if gpt_4_points == "prompt_v1_point_assessment_gpt4-points-run1":
                identifier_to_save_the_result = identifier_to_save_the_result + '_run1'
            elif gpt_4_points == "prompt_v1_point_assessment_gpt4-points-run2":
                identifier_to_save_the_result = identifier_to_save_the_result + '_run2'
        else:
            pass
        
        raters_for_points = [f"melted_{gpt_4_points}", 'human1_points', 'human2_points', 'human3_points']
            
        if per_question_irr == False:  # just the main results


            base_dir = "newly_generated_data_results" if newly_generated_data_gpt4 else "original_data_results"
            calculate_statistics_and_save(df=working_df_points_processed_ready_for_cronbachs_alpha,
                                                        identifier=f"{identifier_to_save_the_result}_result_questions1to6", 
                                                        raters=raters_for_points,
                                                        location_to_save = f"./{base_dir}/point-assessment/pooled/{identifier_to_save_the_result}/{identifier_to_save_the_result}_result_questions1to6.json"  
                                                    )    
            
        elif per_question_irr == True: # per question IRR and main results
            
            
            for question_counter in range(1, 7):
                
                base_dir = "newly_generated_data_results" if newly_generated_data_gpt4 else "original_data_results"
                
                calculate_statistics_and_save(df=working_df_points_processed_ready_for_cronbachs_alpha[working_df_points_processed_ready_for_cronbachs_alpha['Questions'] == f'Question{question_counter}'],
                                        identifier=f"{identifier_to_save_the_result}_result_question{question_counter}",
                                        raters=raters_for_points,
                                        location_to_save = f"./{base_dir}/point-assessment/per-question/{identifier_to_save_the_result}/{identifier_to_save_the_result}_result_questions_{question_counter}.json"
                                        )
                    
                    
                    
                    
                    
            


def main():
    
    parser = argparse.ArgumentParser(description="""
                                     
Developer: Abdullah Al Zubaer
Email: abdullahal.zubaer@uni-passau.de
Institution: University of Passau
Project Page: https://www.uni-passau.de/deepwrite
                                     
Accompanying code for the paper titled: "Can GPT-4 Replace Human Examiners? A
Competition on Checking Open-Text Answers"

Team allocation:
AI-human team 1 = HE-3   & HE-2   & GPT-4
AI-human team 2 = HE-3   & GPT-4  & HE-1
AI-human team 3 = GPT-4  & HE-2   & HE-1

                                  
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    
    parser.add_argument("-nd", "--newly_generated_data_gpt4",
                        action="store_true",
                        help="""If this flag used, this will calculate the rank
                        and point assessments for the newly created data by GPT-4.
                        Use this option when you want to calculate the rank and point
                        assessments for the newly generated data by GPT-4. (default: %(default)s)
                        """)
    
    parser.add_argument("-ndd", "--newly_generated_data_directory",
                        help="""Newly generated data directory by GPT-4 (default: %(default)s""",
                        default="./newly_generated_data/data_gpt4_ranks_points.csv",
                        required=False)
    
    parser.add_argument("-d", "--data",
                        help="Orignal data file from zenodo (default: %(default)s)",
                        default="./original_data/Data_complete_Can_GPT_Replace_Human_Examiners.xlsx",
                        required=False)
    
    parser.add_argument("-at", "--assessment_type",
                        help="Assessment type(s) 'rank' or 'point', or 'rank point' (default: %(default)s)\nRequired",
                        required=True,
                        nargs='+', # One or more arguments
                        )
    
    parser.add_argument("-pq", "--per_question_irr",
                        help="""If this flag used, this will calculate the inter-rater reliability (IRR) per question 
                        for all rank and point assessments.
                        Use this option when you want detailed IRR results for each question.
                        Without this flag, the script will only calculate the pooled results for rank and point assessment.
                        (default: %(default)s)""",
                        action="store_true")

    
    parser.add_argument("-q", "--quiet",
                        help="If this flag used: Suppress all output from the script. Use this option if you prefer a silent run without any console messages.",
                        action="store_true")
    
    args = parser.parse_args()

    if args.quiet:
        # Override the print function
        builtins.print = lambda *args, **kwargs: None
    
    
    
    if not args.newly_generated_data_gpt4:
        # Read from the original data file
        data = read_excel_sheets(file_path=args.data,
                             sheets='Robustness & Extensions')
    else:
        # Read from the  generated data by GPT-4
        data = pd.read_csv(args.newly_generated_data_directory)
        
    # Check if 'rank' is in the list of assessment types
    if 'rank' in args.assessment_type:
        rank_assessment(data=data,
                        per_question_irr=args.per_question_irr,
                        newly_generated_data_gpt4=args.newly_generated_data_gpt4)

    # Check if 'point' is in the list of assessment types
    if 'point' in args.assessment_type:
        point_assessment(data=data,
                         per_question_irr=args.per_question_irr,
                         newly_generated_data_gpt4=args.newly_generated_data_gpt4)

    # Raise an error if neither 'rank' nor 'point' are in the list
    if not {'rank', 'point'} & set(args.assessment_type):
        raise ValueError("Please provide a valid assessment type 'rank' or 'point'")

        
    
if __name__ == "__main__":
    exit(main())
     
