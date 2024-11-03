import time
from pathlib import Path
import torch
import onnx
from onnxsim import simplify
from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, SegmentationModel
from utils.general import LOGGER, colorstr, check_img_size

def export_onnx(model, inputs, file, opset, dynamic, simp, prefix=colorstr("ONNX:")):
    f = str(file.with_suffix(".onnx"))
    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    torch.onnx.export(
        model,
        inputs,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["inputs"],
        output_names=output_names,
        dynamic_axes=None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {"stride": int(max(model.stride)), "names": model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simp:
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, f, save_as_external_data=True, all_tensors_to_one_file=True)
    LOGGER.info(f"onnx path: {f}")

def run():
    t = time.time()
    data= "./data/coco128.yaml" # 'dataset.yaml path'
    weights= "./weights/yolov5s.pt" # weights path
    imgsz=(640, 640) # image (height, width)
    onnx = 1
    simp = True
    opset=12
    batch_size=1
    half=False
    file = Path(weights)  # PyTorch weights

    # Load PyTorch model
    device = torch.device("cpu")
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    inputs = torch.zeros((batch_size, 3, *imgsz)).to(device)  # image size(1,3,320,192) BCHW iDetection
    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True

    for _ in range(2):
        y = model(inputs)  # dry runs
    if half:
        inputs, model = inputs.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata

    # Exports
    export_onnx(model, inputs, file, opset, False, simp)
    

if __name__ == "__main__":
    run()
