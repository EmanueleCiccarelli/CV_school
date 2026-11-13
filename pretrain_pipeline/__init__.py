"""
pretrain_pipeline - Pipeline autonoma di pre-training per VGG16-BN + Multi-Scale Detect Head
(100% compatibile con Ultralytics Platform e CLI yolo train)
"""

from .model import (
    get_vgg16_detector_yaml,
    build_detector_model,
    freeze_batchnorm_layers,
    configure_detector_stage,
)
from .export_checkpoint import prepare_custom_class_checkpoint
