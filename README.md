# Can-GPT-4-Replace-Human-Examiners-A-Competition-on-Checking-Open-Text-Answers

Accompanying code for the paper "Can GPT-4 Replace Human Examiners? A Competition on Checking Open-Text Answers" by the authors:

    Zubaer, Abdullah Al; Granitzer, Michael;
    Geschwind, Stephan; Graf Lambsdorff, Johann; Voss, Deborah

    Affiliation (all authors):  University of Passau, Passau, Germany.


Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11085379.svg)](https://doi.org/10.5281/zenodo.11085379)

# Inter-Rater Reliability Evaluation

This repository includes Python scripts designed to evaluate Inter-Rater Reliability (IRR) using data from specific files, saving the results in a designated folder. Additionally, it provides scripts to generate rank and point assessments using GPT-4 for all the prompts mentioned in our paper.

It consists of two Python scripts:
1. `create_rank_point_gpt4.py`: Generates rank and point data using GPT-4 models.
2. `rank_point_assessment.py`: Evaluates IRR based on generated data or original data.

<!-- ## Features

- Evaluate IRR based on assessment types: 'rank' and/or 'point'.
- Optionally compute IRR per question for detailed insights.
- Create ranks and points using GPT-4 for all the prompts.
- Suppress console output for a silent run. -->


## Installation
Tested on Ubuntu 22.04.4 LTS.

 >To set up the project, you will need to have [Anaconda](https://www.anaconda.com/) installed on your system. If you don't have it installed, you can download it from [here](https://www.anaconda.com/download/success).


 > For setting up OpenAI GPT-4 API, you need to have an API key. You can get it from [here](https://beta.openai.com/signup/) and to set up the API key as environment variable, you can follow the instructions from [here](https://mkyong.com/linux/how-to-set-environment-variable-in-ubuntu/). The key must be set up in helper_functions.py file like this, `openai.api_key = os.getenv("OPENAI_API_KEY")`

 1. Create a conda environment

      ```bash
      conda create -n <env_name> python=3.10
      conda activate <env_name>
      ```

2. **Clone the Repository**: 
   ```bash
   git clone https://github.com/abdullahalzubaer/Can-GPT-4-Replace-Human-Examiners-A-Competition-on-Checking-Open-Text-Answers.git

   cd Can-GPT-4-Replace-Human-Examiners-A-Competition-on-Checking-Open-Text-Answers
   ```



3. **Install Dependencies**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```



   

#  Usage

The two scripts are used sequentially or independently, depending on the data you have. Below are instructions for each scenario.

## 1. Working with Original Data

If you want to evaluate the original data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11085379.svg)](https://doi.org/10.5281/zenodo.11085379) directly without generating any new ranks or points using GPT-4, use the following command:
>The data must be present in `./original_data/` directory.

### 1.1 Rank and Point Assessment for Per-Question IRR:

```bash
python rank_point_assessment.py -at rank point -pq
```
This will calculate the rank and point assessments for per-question IRR and save results in `./original_data_results/...` directories.

### 1.2 Rank and Point Assessment Without Per-Question IRR:
```bash
python rank_point_assessment.py -at rank point
```
This calculates the rank and point assessments and saves pooled results in `./original_data_results/...` directories.




## 2. Generating and Evaluating New Ranks and Points Using GPT-4.

To generate new ranks and points using GPT-4, follow these steps:

### 2.1 Run the script to generate ranks and points:
```bash
python create_rank_point_gpt4.py
```
The generated data will be saved in the `./newly_generated_data/` directory.
>Note: In rare instances, the GPT-4-generated data may not be perfectly parseable due to the non-deterministic nature of the model. You might encounter parsing issues, which can result in incomplete data. In such cases, it might be necessary to manually inspect and modify the data by reviewing the metadata file `metadata_gpt4_ranks_points` to correct the rank and point information in the `data_gpt4_ranks_points` file.

### 2.2 Evaluate the generated data using the rank and point assessment script:

#### 2.2.1 Rank and Point Assessment with Per-Question IRR:

```bash
python rank_point_assessment.py -at rank point -pq -nd
```
This will calculate the rank and point assessments for per-question IRR and save results. in `./newly_generated_data_results/...`

#### 2.2.2 Rank and Point Assessment Without Per-Question IRR:
```bash
python rank_point_assessment.py -at rank point -nd
```
This calculates the rank and point assessments and saves pooled results in `./newly_generated_data_results/...`




<!-- # Arguments

`create_rank_point_gpt4.py`

```bash
python create-rank-point-gpt4.py [-d <data_file>] [-m <model_name>] [-t <temperature>] [-tp <top_p>] [--data_to_save_path <path>] [--metadata_to_save_path <path>] [-q]
```
Arguments
```
    -d, --data: Path to the input data file. Default is ./data/Data_complete_Can_GPT_Replace_Human_Examiners.xlsx.
    -m, --model_name: Name of the OpenAI API model to use. Default is gpt-4-0125-preview.
    -t, --temperature: Temperature parameter for the API. Default is 0.
    -tp, --top_p: Top-p parameter for the API. Default is 1.
    --data_to_save_path: Path to save the generated data. Default is ./newly_generated_data/data_gpt4_ranks_points.
    --metadata_to_save_path: Path to save the metadata. Default is ./newly_generated_data/metadata_gpt4_ranks_points.
    -q, --quiet: Use this flag to suppress all output from the script.
```

`rank_point_assessment.py`

```bash
python rank_point_assessment.py -at <assessment_type> [-d <data_file>] [-pq] [-q] [-nd] [-ndd <newly_generated_data_directory>]
```
Arguments
```
    -nd, --newly_generated_data_gpt4: Use this flag to evaluate the newly generated GPT-4 data.
    -ndd, --newly_generated_data_directory: Directory containing newly generated GPT-4 data. Default is ./newly_generated_data/data_gpt4_ranks_points.csv.
    -d, --data: Path to the original data file. Default is ./original_data/Data_complete_Can_GPT_Replace_Human_Examiners.xlsx.
    -at, --assessment_type: Assessment type(s). Must be one or more of 'rank', 'point', or 'rank point'. This is a required argument.
    -pq, --per_question_irr: Use this flag to calculate the inter-rater reliability per question.
    -q, --quiet: Use this flag to suppress all output from the script.
``` -->









License

This code is licensed under the Apache-2.0 license. See the LICENSE file for details.
Dataset licence mentioned here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11085379.svg)](https://doi.org/10.5281/zenodo.11085379)


# Citation

If you use our dataset or code, please cite the data source and our paper. Proper citation helps to ensure continued support for the project and acknowledges the work of the authors.

Dataset Citation:

```
@dataset{zubaer_2024_11085379,
  author       = {Zubaer, Abdullah Al and
                  Granitzer, Michael and
                  Geschwind, Stephan and
                  Graf Lambsdorff, Johann and
                  Voss, Deborah},
  title        = {{Can GPT-4 Replace Human Examiners?  A Competition 
                   on Checking Open-Text Answers}},
  month        = apr,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.11085379},
  url          = {https://doi.org/10.5281/zenodo.11085379}
}
```


Paper Citation:

```
It will be announced after publication.
```
