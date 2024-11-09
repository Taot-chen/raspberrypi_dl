import argparse
import sys
import time
import numpy as np
from pathlib import Path
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import torch
import torch.nn as nn
import models
from models.common import NMS, NMS_Export
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from onnxsim import simplify

def export_onnx(model, inputs, file, opset, simp, output_names = ["output0", "output1"]):
    try:
        import onnx
        f = str(file.with_suffix(".onnx"))
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        torch.onnx.export(
            model,
            inputs,
            f,
            verbose=False,
            opset_version=opset,
            input_names=['inputs'],
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

        # print(onnx.helper.printable_graph(model_onnx.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). ' % (time.time() - t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt', help='weights path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[320, 320], help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--half', type=int, default=0, help='convert datatype to float16')
    parser.add_argument('--opset', type=int, default=12, help='onnx opset number')
    parser.add_argument('--onnxsim', type=int, default=1, help='simplify onnx')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    inputs = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)

    # Update model
    for k, m in model.named_modules():
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, models.yolo.Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True
            m.forward

    for _ in range(2):
        y = model(inputs)  # dry runs

    if opt.half:
        inputs, model = inputs.half(), model.half()  # to FP16

    # model output shape
    shape = None
    if isinstance(y, tuple):
        shape = tuple(y[0])
    elif isinstance(y, list):
        tuple(y[0].shape)
    else:
        shape = tuple(y.shape)
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    
    # Exports
    export_onnx(model, inputs, Path(opt.weights), opt.opset, simp=opt.onnxsim)
