from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
print("before import dataset")
from tutorial_dataset import MyDataset
print("after import dataset")
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd15_danbooru.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
print("before dataset instance")
dataset = MyDataset()
print("dataset length: ")
print(len(dataset))
print("after dataset instance")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# TODO add checkpoint_callback
checkpoint_callback = ModelCheckpoint(
                dirpath="models_checkpoints/danbooruSmall/",
                every_n_train_steps=2500,
                save_weights_only=False,
                save_top_k = -1
            )

trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger,checkpoint_callback])


# Train!
trainer.fit(model, dataloader)
