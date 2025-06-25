# Example patch script for injecting CBAM into YOLO backbone

from utils.mod_pt_model import CBAM
from ultralytics.nn.modules.conv import Conv

def patch_model_with_cbam(model):
    backbone = model.model if hasattr(model, 'model') else model
    for i, module in enumerate(backbone):
        if isinstance(module, nn.Sequential):
            for j, m in enumerate(module):
                if isinstance(m, Conv) and m.conv.out_channels >= 64:
                    cbam = CBAM(m.conv.out_channels)
                    module.insert(j + 1, cbam)
    return model
