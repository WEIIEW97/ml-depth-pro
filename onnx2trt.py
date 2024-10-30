import tensorrt as trt


def convert_onnx2trt(onnx_model_path: str, trt_model_path: str, is_fp16: bool = False):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse the ONNX model.")

    config = builder.create_builder_config()
    if is_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.FP32)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the engine.")

    with open(trt_model_path, "wb") as f:
        f.write(serialized_engine)

    print(f">>> tensorrt engine working done!")


if __name__ == "__main__":
    onnx_path = "/home/william/Codes/ml-depth-pro/onnx_exp/depth_pro.onnx"
    trt_path = "/home/william/Codes/ml-depth-pro/trt/depth_pro.trt"

    convert_onnx2trt(onnx_path, trt_path)