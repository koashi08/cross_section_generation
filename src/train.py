"""学習ループ本体（Trainer）。

機能:
  - 基本ループ（epoch -> batch -> forward -> loss -> backward -> step）
  - β warmup（z_global / z_var 個別）
  - 検証ループ（実スケール再構成誤差で評価）
  - チェックポイント（best モデル保存）
  - early stopping（val 実スケール再構成誤差で判断）
  - 勾配クリッピング
  - 学習率スケジューラ（ReduceLROnPlateau）
  - ログ記録（dict + CSV）
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict

import torch
from torch_geometric.loader import DataLoader

from .vae import vae_loss
from .metrics import reconstruction_metrics


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    warmup_epochs: int = 10
    beta_global_max: float = 1.0
    beta_var_max: float = 1.0
    free_bits: float = 0.0       # posterior collapse 対策（次元あたり nats）
    grad_clip: float = 1.0
    patience: int = 15                  # early stopping
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    checkpoint_dir: str = "./checkpoints"
    device: str = "cpu"
    log_every: int = 1                  # 何エポックごとにログ表示するか


class Trainer:
    def __init__(self, model, train_ds, val_ds, normalizer, config: TrainConfig):
        self.model = model.to(config.device)
        self.normalizer = normalizer
        self.cfg = config
        self.device = config.device

        self.train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = None
        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min",
                factor=config.scheduler_factor, patience=config.scheduler_patience,
                min_lr=config.min_lr,
            )

        # normalizer の統計をデバイスに移動（loss/metric で使うため）
        self._move_normalizer_to_device()

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.history = []   # 各エポックのログ dict

    def _move_normalizer_to_device(self):
        for attr in ["node_mean", "node_std", "edge_mean", "edge_std",
                     "offset_mean", "offset_std"]:
            v = getattr(self.normalizer, attr, None)
            if v is not None:
                setattr(self.normalizer, attr, v.to(self.device))

    # ------------------------------------------------------------------
    def _compute_beta(self, epoch):
        """β warmup: 0 -> max を warmup_epochs かけて線形に上げる。"""
        frac = min(1.0, (epoch + 1) / max(1, self.cfg.warmup_epochs))
        return self.cfg.beta_global_max * frac, self.cfg.beta_var_max * frac

    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        beta_g, beta_v = self._compute_beta(epoch)
        agg = {"recon": 0.0, "kl_global": 0.0, "kl_var": 0.0, "total": 0.0, "n": 0}

        for batch in self.train_loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            loss, logs = vae_loss(out, batch, normalizer=self.normalizer,
                                  beta_global=beta_g, beta_var=beta_v,
                                  free_bits=self.cfg.free_bits)
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            bs = int(batch.batch.max()) + 1
            for k in ("recon", "kl_global", "kl_var", "total"):
                agg[k] += logs[k] * bs
            agg["n"] += bs

        n = max(agg["n"], 1)
        return {k: agg[k] / n for k in ("recon", "kl_global", "kl_var", "total")}, (beta_g, beta_v)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        beta_g, beta_v = self._compute_beta(epoch)
        agg = {"recon": 0.0, "total": 0.0,
               "mean_euclidean": 0.0, "mse_real": 0.0, "n": 0}

        for batch in self.val_loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            _, logs = vae_loss(out, batch, normalizer=self.normalizer,
                               beta_global=beta_g, beta_var=beta_v,
                               free_bits=self.cfg.free_bits)
            metrics = reconstruction_metrics(out, batch, normalizer=self.normalizer)

            bs = int(batch.batch.max()) + 1
            agg["recon"] += logs["recon"] * bs
            agg["total"] += logs["total"] * bs
            agg["mean_euclidean"] += metrics["mean_euclidean"] * bs
            agg["mse_real"] += metrics["mse_real"] * bs
            agg["n"] += bs

        n = max(agg["n"], 1)
        return {k: agg[k] / n for k in ("recon", "total", "mean_euclidean", "mse_real")}

    # ------------------------------------------------------------------
    def train(self):
        for epoch in range(self.cfg.epochs):
            train_logs, (beta_g, beta_v) = self.train_epoch(epoch)
            val_logs = self.validate(epoch)

            # early stopping / checkpoint は「val 実スケール再構成誤差」で判断
            monitor = val_logs["mean_euclidean"]
            if self.scheduler is not None:
                self.scheduler.step(monitor)
            cur_lr = self.optimizer.param_groups[0]["lr"]

            improved = monitor < self.best_metric - 1e-6
            if improved:
                self.best_metric = monitor
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint("best.pt", epoch, val_logs)
            else:
                self.epochs_no_improve += 1

            record = {
                "epoch": epoch,
                "beta_global": beta_g, "beta_var": beta_v, "lr": cur_lr,
                "train_recon": train_logs["recon"],
                "train_kl_global": train_logs["kl_global"],
                "train_kl_var": train_logs["kl_var"],
                "val_recon": val_logs["recon"],
                "val_mean_euclidean": val_logs["mean_euclidean"],
                "val_mse_real": val_logs["mse_real"],
                "improved": improved,
            }
            self.history.append(record)

            if epoch % self.cfg.log_every == 0 or improved:
                star = " *" if improved else ""
                print(f"[{epoch:3d}] train_recon={train_logs['recon']:.4f} "
                      f"kl_g={train_logs['kl_global']:.3f} kl_v={train_logs['kl_var']:.3f} | "
                      f"val_euclid={val_logs['mean_euclidean']:.4f} "
                      f"(β_g={beta_g:.2f} β_v={beta_v:.2f} lr={cur_lr:.1e}){star}")

            if self.epochs_no_improve >= self.cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best epoch {self.best_epoch}, "
                      f"best val_euclid={self.best_metric:.4f})")
                break

        self.save_history_csv("history.csv")
        return self.history

    # ------------------------------------------------------------------
    def save_checkpoint(self, name, epoch, val_logs):
        path = os.path.join(self.cfg.checkpoint_dir, name)
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_logs": val_logs,
            "config": asdict(self.cfg),
        }, path)

    def load_best(self):
        path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
        ckpt = torch.load(path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        return ckpt

    def save_history_csv(self, name):
        if not self.history:
            return
        path = os.path.join(self.cfg.checkpoint_dir, name)
        keys = list(self.history[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
