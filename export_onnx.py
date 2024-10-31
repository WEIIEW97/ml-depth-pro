import torch
import torch.onnx
import depth_pro

from onnx import load_model, save_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

import cv2

IMAGE_SIZE = 1536


def export_depth_pro_onnx(image_path, onnx_path):
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    image = torch.nn.functional.interpolate(
        image,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    inverse_depth, fov_deg = model(image)  # Use model as callable

    torch.onnx.export(
        model,
        image,
        onnx_path,
        opset_version=19,
        input_names=["pixel_values"],
        output_names=["canonical_inverse_depth", "fov_deg"],
        verbose=False,
        do_constant_folding=True,
    )

    # Optionally use shape inference if needed
    # save_model(
    #     SymbolicShapeInference.infer_shapes(load_model(onnx_path), auto_merge=True),
    #     onnx_path,
    # )

    print(f"Model exported to {onnx_path}")



if __name__ == "__main__":
    # image_path = "/home/william/Codes/ml-depth-pro/data/example.jpg"
    # onnx_path = "/home/william/Codes/ml-depth-pro/onnx_exp/depth_pro.onnx"

    image_path = r"D:\william\codes\ml-depth-pro\data\example.jpg"
    onnx_path = r"D:\william\codes\ml-depth-pro\onnx_exp\depth_pro.onnx"

    export_depth_pro_onnx(image_path, onnx_path)
