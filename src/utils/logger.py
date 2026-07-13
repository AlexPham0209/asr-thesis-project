from transformers import TrainerState
from transformers import TrainerCallback
import logging

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger 

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        current_step = state.global_step
        metrics = " - ".join([f"{key}: {value}" for key, value in logs.items()])
        
        if self.logger is not None:
            self.logger.info(f"step {current_step} - {metrics}")
