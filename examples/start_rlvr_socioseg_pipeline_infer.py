import os
os.environ.setdefault("HF_HOME", os.environ.get("EVEREST_HF_HOME", "../data"))
import argparse

from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_socioseg_vlm_pipeline_infer import SocioSegConfig, SocioSegInferPipeline
from roll.utils.infer_output import build_infer_output_paths


def configure_infer_output_dirs(cfg):
    output_paths = build_infer_output_paths(
        pretrain=cfg.pretrain,
        base_output_dir=cfg.output_dir,
    )
    cfg.output_dir = output_paths["output_dir"]
    cfg.logging_dir = output_paths["logging_dir"]
    cfg.checkpoint_config.output_dir = output_paths["checkpoint_dir"]
    cfg.tracker_kwargs.log_dir = output_paths["tensorboard_dir"]
    OmegaConf.update(cfg, "profiler_output_dir", output_paths["profiler_dir"], force_add=True)
    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="./infer")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="rlvr_megatron"
    )
    args = parser.parse_args()

    if GlobalHydra.instance().is_initialized():
        print("Hydra has been initialized. Now clearing it.")
        GlobalHydra.instance().clear()

    initialize(config_path=args.config_path, job_name="app")

    cfg = compose(config_name=args.config_name)

    output_paths = configure_infer_output_dirs(cfg)
    print(f"Inference artifacts will be saved under: {output_paths['output_dir']}")

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=SocioSegConfig, data=OmegaConf.to_container(cfg, resolve=True))


    init()

    pipeline = SocioSegInferPipeline(pipeline_config=ppo_config)
    print('end init')
    pipeline.run()


if __name__ == "__main__":
    main()
