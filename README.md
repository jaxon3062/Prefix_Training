# Prefix_Training

## Environment Setup

```bash
pip install -r requirement.txt
```

## Usage

1. Set up the correct arguments in `train.py`

```python
# Configuration
model_name = "..."  # Replace with model name
dataset_path = "..."  # Replace with file path
output_dir = "..."  # Set output dir
apply_custom_prompt = True  # Set to False to use default prompt format
use_structure_tuning = True  # Set to True to enable structure tuning
task_template_ratio = 0.1  # Ratio of data that will use the "task" template
prefix_length = 20  # Length of the prefix tuning strings (in characters)

# Define prompt templates
# (Add more fields of different prompts for further structure tuning)
custom_prompt_templates = {
    "default": "{question}\n",
    "task": "{question} Please provide the initial step towards resolving the question. This step may serve as a foundation but might not encompass the entire solution.\n",
    ...
}


```

2. Run the script

```bash
python train.py
```
