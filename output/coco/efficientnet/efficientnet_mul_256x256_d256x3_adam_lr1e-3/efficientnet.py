from models.efficientnet_pytorch import EfficientNet
#先装包
# 调用预训练模型
# 赋值给当前模型
# 保留
def get_pose_net(cfg,is_train, **kwargs):
    model = EfficientNet.from_pretrained(cfg.MODEL.EXTRA.NETTYPE,cfg,**kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:#这个参数在default里
        model.init_weights(cfg.MODEL.EXTRA.DECONV_PRETRAINED)  # 类的函数要调用了才执行

    return model