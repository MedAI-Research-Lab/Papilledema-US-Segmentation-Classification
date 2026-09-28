"""Lazy registry: all model families use the same image-only binary contract."""
from importlib import import_module

REGISTRY = {
    "yolo26": ("yolo_joint", "YOLO26Joint"),
    "vit_method2": ("vit_method2", "ViTMethod2Joint"),
    "emcad": ("emcad_joint", "EMCADJoint"),
    "sam2_unet": ("sam2_unet", "SAM2UNetJoint"),
}


def build_model(name, pretrained=True, image_size=768):
    module, cls = REGISTRY[name]
    if name == "vit_method2":
        pretrained = False  # No external checkpoint exists for these custom models.
    return getattr(import_module(f".{module}", __package__), cls)(pretrained=pretrained, image_size=image_size)
