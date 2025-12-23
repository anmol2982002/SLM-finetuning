# Training package
from .prompts import (
    format_training_prompt,
    format_inference_prompt,
    format_few_shot_prompt,
    SYSTEM_PROMPT,
    TRAINING_TEMPLATE,
    INFERENCE_TEMPLATE
)
from .dataset import InstructionDataset, DataCollatorForInstructionTuning
