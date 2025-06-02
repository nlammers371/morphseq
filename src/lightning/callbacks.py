import json, os, subprocess
from pytorch_lightning import Callback
import pickle
from pathlib import Path


class SaveRunMetadata(Callback):
    def __init__(self, data_cfg):
        # build your payload once
        self.index_dict = {
                "train": data_cfg.train_indices,
                "eval":  data_cfg.eval_indices,
                "test":  data_cfg.test_indices,
            }
        # if hasattr(data_cfg, "metric_array"):
        #     self.metric_array = data_cfg.metric_array
        self._written = False

    def on_train_start(self, trainer, pl_module):
        if self._written:
            return
        run_dir = trainer.logger.save_dir        # tb_logs/run_name/version_x
        index_dir  = os.path.join(run_dir, "split_indices.pkl")
        with open(index_dir, "wb") as file:
            pickle.dump(self.index_dict, file)

        # if hasattr(self, "metric_array"):
        #     metric_dir = os.path.join(run_dir, "metric_array.npy")
        #     self.metric_array = self.metric_array

        self._written = True

class EpochListCheckpoint(Callback):
    def __init__(self, epochs, dirpath="checkpoints"):
        super().__init__()
        self.epochs = set(int(e) for e in epochs)
        self.dir = Path(dirpath)
        self.dir.mkdir(exist_ok=True, parents=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # +1 because current_epoch is 0-based
        epoch = trainer.current_epoch + 1
        if epoch in self.epochs:
            ckpt_path = self.dir / f"epoch{epoch:04d}.ckpt"
            trainer.save_checkpoint(ckpt_path)