import json
import cv2
import numpy as np

print("starting...")
from torch.utils.data import Dataset
print("ending...")

dataset_name = "danbooru/"

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/' + dataset_name + 'prompt.json', 'rt') as f:
            # for line in f:
            #     self.data.append(json.loads(line))
            lines = f.readlines()
            for line in lines:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/' + dataset_name + source_filename)
        target = cv2.imread('./training/' + dataset_name + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        # TODO 如果是 danbooru 数据集，则将控制图（线稿）进行反色处理
        if ("danbooru" in dataset_name):
            source = 255 - cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        else:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

