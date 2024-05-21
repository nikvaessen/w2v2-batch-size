########################################################################################
#
# Entrypoint for training a SSL model.
#
# Author(s): anon
########################################################################################

import hydra

from dotenv import load_dotenv
from omegaconf import OmegaConf

from nanow2v2.other.hydra_resolvers import get_run_id

########################################################################################
# script which should be run


@hydra.main(config_path="config", config_name="ssl", version_base="1.3")
def main(cfg):
    # we import here such that tab-completion in bash
    # does not need to import everything (which slows it down
    # significantly)
    from nanow2v2.train_ssl import train_ssl
    import os

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        import time
        import random

        random.seed(os.environ["SLURM_ARRAY_TASK_ID"])
        time.sleep(random.random() * 3)

    return train_ssl(cfg)


########################################################################################
# execute hydra application

if __name__ == "__main__":
    # enable math operations in config variables
    OmegaConf.register_new_resolver("eval", eval)

    # dynamically decide on $PWD of experiment
    OmegaConf.register_new_resolver("get_run_id", get_run_id)

    load_dotenv()
    main()
