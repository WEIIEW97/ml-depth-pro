from PIL import Image
import depth_pro
import torch


if __name__ == "__main__":
    # print(torch.__version__)
    # print(torch.__path__)

    image_path = "/home/william/Codes/ml-depth-pro/data/example.jpg"
    # Load model and preprocessing transform
    device = torch.device('cuda:0')
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    1
    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    print(f"focal length is {focallength_px}")
    print(f"min value of depth is {depth.min()}, max value of depth is {depth.max()}")