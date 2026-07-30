"""BUS-only 小样本过拟合门禁数据集。"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BUSOverfitDataset(Dataset):
    """只加载 BUS 和亚型标签，避免无关模态及 tokenizer 干扰门禁。"""

    def __init__(self, root_dir, samples, img_size=224, transform=None):
        self.bus_dir = Path(root_dir) / "data" / "images" / "BUS"
        self.samples = [dict(sample) for sample in samples]
        self.transform = transform or transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(
            self.bus_dir / f"{sample['case_id']}.jpg"
        ).convert("RGB")
        return {
            "case_id": sample["case_id"],
            "bus_img": self.transform(image),
            "subtype_label": sample["subtype_label"],
        }
