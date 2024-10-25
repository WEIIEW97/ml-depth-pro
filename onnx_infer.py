import cv2
import numpy as np
import onnxruntime as ort

IMAGE_SIZE = 1536


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = (
        img.astype(np.float32) / 255.0
    )  # Normalize if your model expects floating-point inputs

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # Change data layout from HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img, h, w


def infer(img_path, onnx_path, f_px=None):
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])

    rgb, h, w = preprocess_image(img_path)

    input_name = session.get_inputs()[0].name  # Should print 'pixel_values'
    input_feed = {input_name: rgb}

    output_names = [
        output.name for output in session.get_outputs()
    ]  # Gets all output names
    print(
        "Expected outputs:", output_names
    )  # Should print ['inverse_depth', 'focal_pixel']

    results = session.run(output_names, input_feed)

    canonical_inverse_depth = results[0]
    fov_deg = results[1]
    # print("Inverse Depth Output:", inverse_depth)
    # print("Focal Pixel Output:", focal_pixel)
    if f_px is None:
        f_px = 0.5 * w / np.tan(0.5 * np.deg2rad(fov_deg.squeeze()))

    inverse_depth = canonical_inverse_depth.squeeze(0).transpose(1, 2, 0)

    print(inverse_depth.shape)
    print(f_px)

    inverse_depth_restore = cv2.resize(inverse_depth, (h, w))

    print(f"full inverse depth is: {inverse_depth_restore}")
    print(f"estimated focal length is: {f_px}")


if __name__ == "__main__":
    img_path = "/home/william/Codes/ml-depth-pro/data/example.jpg"
    onnx_path = "/home/william/Codes/ml-depth-pro/onnx_exp/depth_pro.onnx"

    infer(img_path, onnx_path)
