from torchinfo import summary
from thop import profile,clever_format
import time 
import logging

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    logging.info(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


def quality(model, x):
    # summary(model,(args.in_channel, args.length), device='cuda')
    # summary函数会显示模型参数总数，但不会直接显示模型占用的内存大小（如MB/GB）。
    # 下面代码可以估算模型参数占用的显存（仅参数，不含中间激活/缓存）：
    from io import StringIO
    import sys
    tmp_stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = tmp_stdout
    summary(model, input_size=x.shape)
    sys.stdout = old_stdout
    logging.info(f"\n{tmp_stdout.getvalue()}")

    # 计算模型参数占用的内存（单位：MB）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    logging.info(f"model size is {size_all_mb:.3f} MB")

    start_time = time.time()
    out = model(x)
    end_time = time.time()
    flops, params = profile(model, inputs=(x,))

    # conver results to be legible
    f, p = clever_format([flops, params], '%.3f')
 
    logging.info(f"***************** Detales of model Input shape is {x.shape} **********************")
    logging.info(f"Flops of model: {f}  ({flops}), \n  number of parameters: {p} ({params}), \n Time to calculate: {(end_time-start_time)*1000} ms \n")