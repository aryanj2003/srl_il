import sys
import signal
import traceback
import logging
import hydra
from omegaconf import OmegaConf
import pathlib
import torch
import wandb
from pathlib import Path
import os

# use line-buffering for both stdout and stderr. TODO: check if this is necessary
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)



# Create output directory if it doesn't exist
output_dir = "/home/aryan/IL_Workspace/srl_il/output"
os.makedirs(output_dir, exist_ok=True)

# Set up global logging
logging.basicConfig(
    filename=f"{output_dir}/pipeline_crash.log",
    level=logging.DEBUG,
    format="%(asctime)s [CRASH] %(message)s"
)

# Signal handler for all catchable signals
def global_signal_handler(sig, frame):
    signal_name = signal.Signals(sig).name
    error_msg = f"[CRASH DETECTED] Signal={signal_name}, frame={frame}"
    stack_trace = f"[CRASH DETECTED] Traceback:\n{''.join(traceback.format_stack())}"
    logging.error(error_msg)
    logging.error(stack_trace)
    print(error_msg)
    print(stack_trace)
    sys.exit(1)

# Register handlers for all relevant signals
for sig in [signal.SIGSEGV, signal.SIGTERM, signal.SIGABRT, signal.SIGINT, signal.SIGFPE]:
    signal.signal(sig, global_signal_handler)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('srl_il','cfg'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # create the pipeline
    pipeline_cls = hydra.utils.get_class(cfg.pipeline._target_)

    pipeline = pipeline_cls(**cfg)
    pipeline.run()

if __name__ == "__main__":
    logging.debug("Pipeline started")
    main()
