from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import ImageReward.ImageReward as RM

class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        # self.model_path = "/mnt/afs/intern/fangwenhan/qchen/Reward/ImageReward/ImageReward.pt"  # 20250930: manually
        # self.med_config = "/mnt/afs/intern/fangwenhan/qchen/Reward/ImageReward/med_config.json"
        self.model_path = "../Reward/ImageReward/ImageReward.pt"  # 20250930: manually
        self.med_config = "../Reward/ImageReward/med_config.json"
        self.device = device
        self.dtype = dtype
        self.model = RM.load(self.model_path, device=device, download_root='.', med_config=self.med_config).eval().to(dtype=dtype)
        print("Reward Model loads successfully!")
        self.model.requires_grad_(False)
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        rewards = []
        for prompt,image in zip(prompts, images):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(reward)
        return rewards

# Usage example
def main():
    scorer = ImageRewardScorer(
        device="cuda",
        dtype=torch.float32
    )

    images=[
    "astronaut.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()