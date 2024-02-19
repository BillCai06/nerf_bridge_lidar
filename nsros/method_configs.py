# Code slightly adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/configs/method_configs.py

"""
Method configurations
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig


from nsros.ros_datamanager import (
    ROSDataManagerConfig,
    ROSDataManager,
    ROSFullImageDataManagerConfig,
    ROSFullImageDataManager,
)
from nsros.ros_dataparser import ROSDataParserConfig
from nsros.ros_trainer import ROSTrainerConfig
from nsros.ros_dataset import ROSDataset
from nsros.ros_splatfacto import ROSSplatfactoModelConfig

method_configs: Dict[str, ROSTrainerConfig] = {}
descriptions = {
    "ros_nerfacto": "Run the nerfstudio nerfacto method on data streamed from ROS.",
}

method_configs["ros_nerfacto"] =ROSTrainerConfig(
        method_name="ros-depth-splatfacto",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=VanillaPipelineConfig(
            datamanager=ROSFullImageDataManagerConfig(
                _target=ROSFullImageDataManager[ROSDataset],
                dataparser=ROSDataParserConfig(aabb_scale=1.0),
            ),
            model=ROSSplatfactoModelConfig(num_random=50),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )
 





# ROSTrainerConfig(
#     method_name="ros_nerfacto",
#     steps_per_eval_batch=500,
#     steps_per_save=30000,
#     max_num_iterations=30000,
#     mixed_precision=True,
#     pipeline=VanillaPipelineConfig(
#         datamanager=ROSDataManagerConfig(
#             dataparser=ROSDataParserConfig(
#                 aabb_scale=0.8,
#             ),
#             train_num_rays_per_batch=4096,
#             eval_num_rays_per_batch=4096,
#             camera_optimizer=CameraOptimizerConfig(
#                 mode="SO3xR3",
#                 optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
#             ),
#         ),
#         model=SplatfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
#     ),
#    optimizers={
#         "xyz": {
#             "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
#             "scheduler": None,
#         },
#         "features_dc": {
#             "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
#             "scheduler": None,
#         },
#         "features_rest": {
#             "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
#             "scheduler": None,
#         },
#         "opacity": {
#             "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
#             "scheduler": None,
#         },
#         "scaling": {
#             "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
#             "scheduler": None,
#         },
#         "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
#         "camera_opt": {
#             "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
#             "scheduler": None,
#         },
#     },
#     viewer=ViewerConfig(num_rays_per_chunk=20000),
#     vis="viewer",
# )



# RosDepthSplatfacto = MethodSpecification(
#     config=ROSTrainerConfig(
#         method_name="ros-depth-splatfacto",
#         steps_per_eval_image=100,
#         steps_per_eval_batch=0,
#         steps_per_save=2000,
#         steps_per_eval_all_images=1000,
#         max_num_iterations=30000,
#         mixed_precision=False,
#         gradient_accumulation_steps={"camera_opt": 100},
#         pipeline=VanillaPipelineConfig(
#             datamanager=ROSFullImageDataManagerConfig(
#                 _target=ROSFullImageDataManager[ROSDepthDataset],
#                 dataparser=ROSDataParserConfig(aabb_scale=1.0),
#             ),
#             model=ROSSplatfactoModelConfig(num_random=50),
#         ),
#         optimizers={
#             "xyz": {
#                 "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(
#                     lr_final=1.6e-6,
#                     max_steps=30000,
#                 ),
#             },
#             "features_dc": {
#                 "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
#                 "scheduler": None,
#             },
#             "features_rest": {
#                 "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
#                 "scheduler": None,
#             },
#             "opacity": {
#                 "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
#                 "scheduler": None,
#             },
#             "scaling": {
#                 "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
#                 "scheduler": None,
#             },
#             "rotation": {
#                 "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
#                 "scheduler": None,
#             },
#             "camera_opt": {
#                 "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
#                 "scheduler": ExponentialDecaySchedulerConfig(
#                     lr_final=5e-5, max_steps=30000
#                 ),
#             },
#         },
#         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
#         vis="viewer",
#     ),
#     description="Run NerfBridge with the Splatfacto model, and train with streamed RGB and depth images used to seed new gaussains.",
# )



AnnotatedBaseConfigUnion = (
    tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(
                defaults=method_configs, descriptions=descriptions
            )
        ]
    ]
)
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
