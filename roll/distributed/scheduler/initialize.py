import os
import subprocess
import sys
import tempfile
import time

import ray

from roll.distributed.scheduler.driver_utils import (
    get_driver_rank,
    get_driver_master_addr,
    get_driver_node_name,
    get_driver_master_port,
    get_driver_world_size,
    get_ray_status,
    is_ray_cluster_running,
    wait_for_nodes,
)
from roll.distributed.scheduler.log_monitor import LogMonitorListener
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.utils.ray_utils import RayUtils

logger = get_logger()


def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    node_name = get_driver_node_name()


    if world_size == 1:
        logger.info("Single node detected, skipping Ray cluster CLI startup")
        return False

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized")
        return False

    if rank == 0:
        cmd = f"ray start --head --port={master_port} --node-name={node_name}"
    else:

        time.sleep(5)
        cmd = f"ray start --address={master_addr}:{master_port} --node-name={node_name}"

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True,text=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


def init():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    manual_start = start_ray_cluster()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    custom_env_vars = RayUtils.get_custom_env_env_vars()
    custom_env_vars["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"


    hf_home_raw = os.environ.get("HF_HOME", os.path.join(project_root, "data"))
    custom_env_vars["HF_HOME"] = os.path.abspath(hf_home_raw)
    runtime_env = {
        "env_vars": custom_env_vars,
    }
    if not ray.is_initialized():
        object_store_memory = os.environ.get("RAY_OBJECT_STORE_MEMORY")
        ray_init_kwargs = {}
        if object_store_memory:
            ray_init_kwargs["object_store_memory"] = int(object_store_memory)
        ray_temp_dir = os.environ.get("EVEREST_RAY_TMPDIR", os.path.join(tempfile.gettempdir(), "everest-ray"))
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray.init(

            _temp_dir=ray_temp_dir,
            address=f"{master_addr}:{master_port}" if manual_start else None,
            namespace=RAY_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=not manual_start,
            runtime_env=runtime_env,
            **ray_init_kwargs,
        )
        logger.info("Ray cluster initialized")

    if manual_start:
        wait_for_nodes(expected=world_size)
        listener = LogMonitorListener()
        listener.start()

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")

    if manual_start and rank > 0:
        sys.exit(0)
