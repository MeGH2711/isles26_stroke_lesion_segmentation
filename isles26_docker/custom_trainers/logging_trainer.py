"""
logging_trainer.py

nnUNetTrainerWithLogging - identical to plain nnUNetTrainer in every way
(same loss, same sampling, same epoch count, same everything) EXCEPT it also
writes a per-epoch CSV/JSON log, same format/fields as
nnUNetTrainerInstanceLoss's logging (minus the instance-loss-specific
columns, since there is no instance loss here).

Use this instead of nnUNetTrainerInstanceLoss whenever you want the
convenience of per-epoch training_logs/ files WITHOUT changing anything
else about training - e.g. for the ResEnc architecture comparison, where the
whole point is to isolate the architecture as the only variable.

Place this file at:
    nnunetv2/training/nnUNetTrainer/variants/logging_trainer.py

Run with, e.g.:
    nnUNetv2_train Dataset001_ISLES26 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerWithLogging
"""

import csv
import json
import os

import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWithLogging(nnUNetTrainer):

    def initialize(self):
        super().initialize()

        # --- per-epoch CSV/JSON logging setup (same mechanism as
        # nnUNetTrainerInstanceLoss, minus the instance-loss-specific fields) ---
        self.training_logs_dir = os.path.join(self.output_folder, 'training_logs')
        os.makedirs(self.training_logs_dir, exist_ok=True)
        self._epoch_log_records = []
        self._epoch_log_csv_path = os.path.join(
            self.training_logs_dir, f'training_log_fold{self.fold}.csv')
        self._epoch_log_json_path = os.path.join(
            self.training_logs_dir, f'training_log_fold{self.fold}.json')

    def on_epoch_end(self):
        # Capture the epoch number BEFORE calling super() - nnU-Net's base
        # on_epoch_end() increments self.current_epoch at the very end of its
        # own implementation.
        finished_epoch = self.current_epoch
        super().on_epoch_end()
        self._write_epoch_log(finished_epoch)

    def _find_logging_dict(self):
        """
        Different nnU-Net versions store per-epoch history under different
        attribute names/structures on self.logger. Search self.logger's
        attributes (and one level of nesting) for a dict that actually
        contains 'train_losses' as a key, and use whichever one is found -
        same version-robust approach used in the instance-loss trainer.
        """
        if getattr(self, '_logging_dict_cache', None) is not None:
            return self._logging_dict_cache

        def _scan(obj):
            for _, val in vars(obj).items():
                if isinstance(val, dict) and 'train_losses' in val:
                    return val
            return None

        found = _scan(self.logger)
        if found is None:
            for _, val in vars(self.logger).items():
                if hasattr(val, '__dict__'):
                    found = _scan(val)
                    if found is not None:
                        break

        self._logging_dict_cache = found
        return found

    @staticmethod
    def _to_native(value):
        """Convert numpy scalar types to native Python float/int for json.dump."""
        if value is None:
            return None
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    def _write_epoch_log(self, epoch: int):
        # Must never crash a training run - logging is a nice-to-have.
        try:
            log = self._find_logging_dict()

            def _last(key):
                if log is None:
                    return None
                vals = log.get(key)
                return vals[-1] if vals else None

            ts_start = _last('epoch_start_timestamps')
            ts_end = _last('epoch_end_timestamps')

            record = {
                'epoch': epoch,
                'train_loss': _last('train_losses'),
                'val_loss': _last('val_losses'),
                'mean_fg_dice': _last('mean_fg_dice'),
                'ema_fg_dice': _last('ema_fg_dice'),
                'lr': _last('lrs'),
                'epoch_time_s': (ts_end - ts_start) if (ts_start is not None and ts_end is not None) else None,
            }
            record = {k: self._to_native(v) for k, v in record.items()}

            self._epoch_log_records.append(record)

            os.makedirs(self.training_logs_dir, exist_ok=True)
            write_header = not os.path.exists(self._epoch_log_csv_path)
            with open(self._epoch_log_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

            with open(self._epoch_log_json_path, 'w') as f:
                json.dump(self._epoch_log_records, f, indent=2)

        except Exception as e:
            try:
                self.print_to_log_file(
                    f"[nnUNetTrainerWithLogging] WARNING: per-epoch CSV/JSON "
                    f"logging failed for epoch {epoch}: {repr(e)}. "
                    f"Training continues normally; this only affects the "
                    f"training_logs/ files."
                )
            except Exception:
                pass