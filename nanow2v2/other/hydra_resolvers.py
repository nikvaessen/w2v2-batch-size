def get_run_id(_cache={"run_id": None}) -> str:
    import os

    run_id = _cache["run_id"]

    # check if we've already cached the result
    # (we need to cache because it's determined randomly)
    if run_id is not None:
        return run_id

    # on SLURM, we use SLURM_ARRAY_JOB_ID
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_ARRAY_JOB_ID"]
        task_id = os.environ["SLURM_ARRAY_TASK_ID"]

        run_id = f"{job_id}_{task_id}"

    # OR SLURM_JOB_ID, which will be unique
    elif "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        run_id = f"{job_id}"

    # if we're running locally, it will simply be something like
    # 2023-05-13---12:05:03_{postfix}
    # check if we're in a multi-run
    else:
        from hydra.types import RunMode
        from hydra.core.hydra_config import HydraConfig
        import random
        import string
        import time
        import datetime

        if (hydra_cfg := HydraConfig.get()).mode == RunMode.MULTIRUN:
            postfix = str(hydra_cfg.job.num)
        else:
            postfix = "".join(
                random.Random(time.time_ns()).choices(string.ascii_lowercase, k=4)
            )

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        run_id = f"{date_str}---{time_str}_{postfix}"

    _cache["run_id"] = run_id
    return run_id
