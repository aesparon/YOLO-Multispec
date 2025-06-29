import torch
import torch.nn as nn
from ultralytics import YOLO
#from ultralytics.utils.loss import v8SegLoss
from ultralytics.nn.modules import Conv

# ------------------------ Spectral Convolution ------------------------
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spectral = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.combine = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.spectral(x)
        return self.combine(x)

# ------------------------ CBAM Module ------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid_channel(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x

# ------------------------ ECA Module ------------------------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

# ------------------------ DropBlock Regularization ------------------------
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob * x.numel() / (x.size(0) * x.size(1) * (x.size(2) - self.block_size + 1) * (x.size(3) - self.block_size + 1))
        mask = (torch.rand_like(x[:, 0:1, :, :]) < gamma).float()
        mask = nn.functional.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        return x * (1.0 - mask)

# ------------------------ GroupNorm Function ------------------------
def norm_fn(channels):
    return nn.GroupNorm(4, channels)  # 4 groups (adjust based on C)

# Optional: Monkey patch custom loss function

def bbox_iou(pred, target, iou_type='eiou'):
    # Dummy placeholder for bbox_iou computation
    # You must implement or reuse actual EIoU logic from Ultralytics or elsewhere
    return torch.rand(pred.shape[0], device=pred.device)

def focal_eiou_loss(pred_boxes, target_boxes, pred_cls, target_cls):
    eiou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, iou_type='eiou')
    p = torch.sigmoid(pred_cls)
    alpha = 0.25
    gamma = 2.0
    focal = -alpha * (1 - p) ** gamma * target_cls * torch.log(p + 1e-9)
    return eiou_loss.mean() + focal.mean()

# Patch backbone

def patch_backbone_with_attention(model_nn, use_cbam=False, use_eca=False, use_spectral=False, use_dropblock=False, drop_prob=0.1):
    print("\n🔧 Patching YOLO backbone")
    backbone = model_nn.model if hasattr(model_nn, 'model') else model_nn
    spectral_replaced = False

    for i, module in enumerate(backbone):
        if isinstance(module, nn.Sequential):
            for j, m in enumerate(module):
                if use_spectral and not spectral_replaced and isinstance(m, nn.Module) and hasattr(m, 'conv'):
                    orig_conv = m.conv
                    spectral = SpectralConv(orig_conv.in_channels, orig_conv.out_channels)
                    m.conv = spectral.combine
                    module[j] = nn.Sequential(spectral, m.bn, m.act)
                    spectral_replaced = True
                    print(f"🔁 Replaced first Conv with SpectralConv at block {i}, layer {j}")
                    continue

                if isinstance(m, nn.Module) and hasattr(m, 'conv') and m.conv.out_channels >= 64:
                    insertions = []
                    if use_cbam:
                        insertions.append(CBAM(m.conv.out_channels))
                    if use_eca:
                        insertions.append(ECA(m.conv.out_channels))
                    if use_dropblock:
                        insertions.append(DropBlock2D(drop_prob=drop_prob))

                    for k, block in enumerate(insertions):
                        module.insert(j + 1 + k, block)
                    print(f"✨ Inserted {', '.join([type(b).__name__ for b in insertions])} at block {i}, layer {j + 1}")

    def replace_bn_with_gn(m):
        for name, child in m.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(m, name, norm_fn(child.num_features))
                print(f"🔄 Replaced BatchNorm2d with GroupNorm for layer: {name}")
            else:
                replace_bn_with_gn(child)

    replace_bn_with_gn(model_nn)
    print("✅ YOLO backbone patching complete\n")

# Load and patch model

# def train():
#     model = YOLO('yolov8x-seg.yaml')  # Custom 5-channel yaml
#     patch_backbone_with_attention(model.model)

#     model.train(
#         data='path/to/dataset.yaml',
#         epochs=100,
#         imgsz=1024,
#         device=0,
#         batch=8,
#         workers=4,
#         lr0=0.001,
#         project='multispectral_yolo_train',
#         name='cbam_eca_spectral_bifpn',
#         resume=False,
#         pretrained=False
#     )

# if __name__ == "__main__":
#     train()
