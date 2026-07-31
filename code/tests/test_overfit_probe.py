import torch

from code.models.stage2.overfit_probe import BUSOverfitProbe


def test_overfit_probe_returns_four_class_logits():
    """验证门禁模型输出四分类 logits 与图像特征。"""
    model = BUSOverfitProbe(
        pretrained_path=None,
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
    )
    output = model(torch.randn(4, 1, 32, 32))
    assert output["class_logits"].shape == (4, 4)
    assert output["image_features"].shape == (4, 48)


def test_all_overfit_probe_parameters_receive_finite_gradients():
    """验证所有可训练参数均获得非零有限梯度。"""
    model = BUSOverfitProbe(
        pretrained_path=None,
        img_size=32,
        patch_size=16,
        embed_dim=48,
        depth=2,
        num_heads=4,
    )
    labels = torch.tensor([0, 1, 2, 3])
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(4, 1, 32, 32))["class_logits"],
        labels,
    )
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert all(parameter.grad.abs().sum() > 0 for parameter in trainable)
