"""
logging_trainer.py

nnUNetTrainerWithLogging - identical to plain nnUNetTrainer in every way
(same loss, same sampling, same epoch count, same everything) EXCEPT it also
writes a per-epoch CSV/JSON log.
"""

import csv
import json
import os
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWithLogging(nnUNetTrainer):

    def initialize(self):
        super().initialize()
        self.training_logs_dir = os.path.join(self.output_folder, 'training_logs')
        os.makedirs(self.training_logs_dir, exist_ok=True)
        self._epoch_log_records = []
        self._epoch_log_csv_path = os.path.join(
            self.training_logs_dir, f'training_log_fold{self.fold}.csv')
        self._epoch_log_json_path = os.path.join(
            self.training_logs_dir, f'training_log_fold{self.fold}.json')

    def on_epoch_end(self):
        finished_epoch = self.current_epoch
        super().on_epoch_end()
        self._write_epoch_log(finished_epoch)

    def _find_logging_dict(self):
        candidates = []
        if hasattr(self, 'logger') and self.logger is not None:
            candidates.append(self.logger)
            for attr_name in dir(self.logger):
                if attr_name.startswith('_'):
                    continue
                try:
                    candidates.append(getattr(self.logger, attr_name))
                except Exception:
                    pass

        for c in candidates:
            if isinstance(c, dict) and 'train_losses' in c:
                return c
            if hasattr(c, '__dict__') and isinstance(c.__dict__, dict):
                if 'train_losses' in c.__dict__:
                    return c.__dict__
                for v in c.__dict__.values():
                    if isinstance(v, dict) and 'train_losses' in v:
                        return v
        return None

    def _write_epoch_log(self, epoch: int):
        try:
            log_dict = self._find_logging_dict()
            if log_dict is None:
                return

            def _get(key):
                seq = log_dict.get(key, [])
                if not seq:
                    return None
                if epoch < len(seq):
                    val = seq[epoch]
                else:
                    val = seq[-1]
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = float(np.mean(val))
                elif isinstance(val, (np.floating, np.integer)):
                    val = float(val)
                return val

            train_loss = _get('train_losses')
            val_loss = _get('val_losses')
            pseudo_dice = _get('dice_per_class_or_region')
            ema_dice = _get('ema_fg_dice')
            lr = _get('learning_rates')
            epoch_time = _get('epoch_times')

            record = {
                'epoch': int(epoch),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pseudo_dice': pseudo_dice,
                'ema_fg_dice': ema_dice,
                'learning_rate': lr,
                'epoch_time_seconds': epoch_time,
            }

            self._epoch_log_records.append(record)

            with open(self._epoch_log_json_path, 'w') as f:
                json.dump(self._epoch_log_records, f, indent=2)

            fieldnames = [
                'epoch', 'train_loss', 'val_loss', 'pseudo_dice',
                'ema_fg_dice', 'learning_rate', 'epoch_time_seconds',
            ]
            with open(self._epoch_log_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._epoch_log_records)

        except Exception as e:
            self.print_to_log_file(f"[nnUNetTrainerWithLogging] Warning: Failed to write log: {e}")
