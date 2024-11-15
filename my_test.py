import torch

weight = r'weights/nanodet-plus-m_416.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(weight, map_location=device)
# model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()


tt = r'nanodet/nanodet_model_best.pth'
test = torch.load(tt, map_location=device)