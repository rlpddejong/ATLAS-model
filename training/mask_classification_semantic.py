from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        num_procedures: int = 0,
        procedure_coefficient: float = 1.0,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        phase_contrastive_enabled: bool = False,
        contrastive_coefficient: float = 0.1,
        contrastive_temperature: float = 0.07,
        contrastive_pos_radius: int = 1,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            num_procedures=num_procedures,
            procedure_coefficient=procedure_coefficient,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)
        self.phase_contrastive_enabled = phase_contrastive_enabled
        self.contrastive_coefficient = contrastive_coefficient
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_pos_radius = max(1, int(contrastive_pos_radius))

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

    @staticmethod
    def _flatten_clip_batch(imgs, targets):
        if isinstance(imgs, torch.Tensor) and imgs.ndim == 5:
            bsz, clip_len = imgs.shape[:2]
            flat_imgs = imgs.flatten(0, 1)

            flat_targets = []
            time_indices = []
            clip_ids = []
            for b in range(bsz):
                clip_targets = targets[b]
                for t in range(clip_len):
                    target = clip_targets[t]
                    flat_targets.append(target)
                    time_indices.append(
                        int(target.get("time_index", torch.tensor(t)).item())
                    )
                    clip_ids.append(b)

            return (
                flat_imgs,
                flat_targets,
                torch.tensor(time_indices, device=flat_imgs.device, dtype=torch.long),
                torch.tensor(clip_ids, device=flat_imgs.device, dtype=torch.long),
            )

        bsz = len(targets)
        time_indices = torch.zeros(bsz, device=imgs.device, dtype=torch.long)
        clip_ids = torch.arange(bsz, device=imgs.device, dtype=torch.long)
        return imgs, targets, time_indices, clip_ids

    def _temporal_contrastive_loss(self, embeddings, clip_ids, time_indices):
        if embeddings is None or embeddings.shape[0] < 2:
            return embeddings.new_zeros(()) if embeddings is not None else torch.tensor(0.0, device=self.device)

        logits = embeddings @ embeddings.T
        logits = logits / self.contrastive_temperature

        num = logits.shape[0]
        eye = torch.eye(num, dtype=torch.bool, device=logits.device)

        same_clip = clip_ids[:, None] == clip_ids[None, :]
        time_dist = (time_indices[:, None] - time_indices[None, :]).abs()
        pos_mask = same_clip & (time_dist > 0) & (time_dist <= self.contrastive_pos_radius)

        valid_pair_mask = ~eye
        logits = logits.masked_fill(~valid_pair_mask, float("-inf"))

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0)

        pos_counts = pos_mask.sum(dim=1)
        valid_anchors = pos_counts > 0
        if not valid_anchors.any():
            return embeddings.new_zeros(())

        loss = -(pos_log_prob.sum(dim=1)[valid_anchors] / pos_counts[valid_anchors]).mean()
        return loss

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        if (
            isinstance(imgs, torch.Tensor)
            and imgs.ndim == 5
            and getattr(self.network, "temporal_query_learning_enabled", False)
        ):
            return self._training_step_temporal_query(batch)

        imgs, targets, time_indices, clip_ids = self._flatten_clip_batch(imgs, targets)

        outputs = self(
            imgs,
            time_indices=time_indices,
            return_temporal_embedding=self.phase_contrastive_enabled,
        )

        if len(outputs) == 4:
            (
                mask_logits_per_block,
                class_logits_per_block,
                procedure_logits_per_block,
                temporal_embeddings,
            ) = outputs
        else:
            (
                mask_logits_per_block,
                class_logits_per_block,
                procedure_logits_per_block,
            ) = outputs
            temporal_embeddings = None

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        loss_total = self.criterion.loss_total(losses_all_blocks, self.log)

        if procedure_logits_per_block and "procedure" in targets[0]:
            procedure_logits = procedure_logits_per_block[-1]
            procedure_labels = torch.as_tensor(
                [int(target["procedure"]) for target in targets],
                device=procedure_logits.device,
            )
            procedure_loss = torch.nn.functional.cross_entropy(
                procedure_logits, procedure_labels
            )
            self.log(
                "losses/train_procedure_cross_entropy",
                procedure_loss,
                sync_dist=True,
            )
            loss_total = loss_total + (self.procedure_coefficient * procedure_loss)

        if self.phase_contrastive_enabled:
            contrastive_loss = self._temporal_contrastive_loss(
                temporal_embeddings,
                clip_ids=clip_ids,
                time_indices=time_indices,
            )
            self.log(
                "losses/train_temporal_contrastive",
                contrastive_loss,
                sync_dist=True,
            )
            loss_total = loss_total + (self.contrastive_coefficient * contrastive_loss)

        return loss_total

    def _training_step_temporal_query(self, batch):
        imgs, targets = batch
        bsz, clip_len = imgs.shape[:2]

        losses_all_blocks = {}
        last_procedure_logits = None
        flat_targets = []
        flat_time_indices = []
        flat_clip_ids = []
        temporal_embeddings_per_frame = []
        prev_query_embed = None

        for t in range(clip_len):
            frame_imgs = imgs[:, t, ...]
            frame_targets = [targets[b][t] for b in range(bsz)]
            frame_time_indices = torch.tensor(
                [
                    int(target.get("time_index", torch.tensor(t)).item())
                    for target in frame_targets
                ],
                device=frame_imgs.device,
                dtype=torch.long,
            )

            outputs = self(
                frame_imgs,
                time_indices=frame_time_indices,
                return_temporal_embedding=self.phase_contrastive_enabled,
                prev_query_embed=prev_query_embed,
                return_query_embedding=True,
            )

            if self.phase_contrastive_enabled:
                (
                    mask_logits_per_block,
                    class_logits_per_block,
                    procedure_logits_per_block,
                    temporal_embeddings,
                    current_query_embed,
                ) = outputs
                temporal_embeddings_per_frame.append(temporal_embeddings)
            else:
                (
                    mask_logits_per_block,
                    class_logits_per_block,
                    procedure_logits_per_block,
                    current_query_embed,
                ) = outputs

            prev_query_embed = current_query_embed
            if procedure_logits_per_block:
                last_procedure_logits = procedure_logits_per_block[-1]

            for i, (mask_logits, class_logits) in enumerate(
                list(zip(mask_logits_per_block, class_logits_per_block))
            ):
                losses = self.criterion(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    targets=frame_targets,
                )
                block_postfix = self.block_postfix(i)
                for key, value in losses.items():
                    merged_key = f"{key}{block_postfix}"
                    if merged_key in losses_all_blocks:
                        losses_all_blocks[merged_key] = losses_all_blocks[merged_key] + value
                    else:
                        losses_all_blocks[merged_key] = value

            flat_targets.extend(frame_targets)
            flat_time_indices.append(frame_time_indices)
            flat_clip_ids.append(
                torch.arange(bsz, device=frame_imgs.device, dtype=torch.long)
            )

        losses_all_blocks = {
            key: value / clip_len for key, value in losses_all_blocks.items()
        }
        loss_total = self.criterion.loss_total(losses_all_blocks, self.log)

        if last_procedure_logits is not None and "procedure" in flat_targets[0]:
            procedure_labels = torch.as_tensor(
                [int(targets[b][0]["procedure"]) for b in range(bsz)],
                device=last_procedure_logits.device,
            )
            procedure_loss = torch.nn.functional.cross_entropy(
                last_procedure_logits, procedure_labels
            )
            self.log(
                "losses/train_procedure_cross_entropy",
                procedure_loss,
                sync_dist=True,
            )
            loss_total = loss_total + (self.procedure_coefficient * procedure_loss)

        if self.phase_contrastive_enabled:
            temporal_embeddings = torch.cat(temporal_embeddings_per_frame, dim=0)
            clip_ids = torch.cat(flat_clip_ids, dim=0)
            time_indices = torch.cat(flat_time_indices, dim=0)
            contrastive_loss = self._temporal_contrastive_loss(
                temporal_embeddings,
                clip_ids=clip_ids,
                time_indices=time_indices,
            )
            self.log(
                "losses/train_temporal_contrastive",
                contrastive_loss,
                sync_dist=True,
            )
            loss_total = loss_total + (self.contrastive_coefficient * contrastive_loss)

        return loss_total

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        if (
            getattr(self.network, "temporal_query_learning_enabled", False)
            and len(imgs) > 0
            and isinstance(imgs[0], torch.Tensor)
            and imgs[0].ndim == 4
        ):
            return self._eval_step_temporal_query(batch, batch_idx, log_prefix)

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        (
            mask_logits_per_layer,
            class_logits_per_layer,
            procedure_logits_per_layer,
        ) = self(crops)

        self.log_procedure_accuracy(procedure_logits_per_layer, targets, log_prefix, origins)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def _eval_step_temporal_query(self, batch, batch_idx=None, log_prefix=None):
        imgs, targets = batch
        bsz = len(imgs)
        clip_len = imgs[0].shape[0]
        prev_query_embed = None

        for t in range(clip_len):
            frame_imgs = [imgs[b][t] for b in range(bsz)]
            frame_targets = [targets[b][t] for b in range(bsz)]

            img_sizes = [img.shape[-2:] for img in frame_imgs]
            crops, origins = self.window_imgs_semantic(frame_imgs)

            time_indices = torch.tensor(
                [
                    int(target.get("time_index", torch.tensor(t)).item())
                    for target in frame_targets
                ],
                device=crops.device,
                dtype=torch.long,
            )
            crop_time_indices = torch.stack(
                [time_indices[img_i] for img_i, _, _ in origins],
                dim=0,
            )

            crop_prev_query_embed = None
            if prev_query_embed is not None:
                crop_prev_query_embed = torch.stack(
                    [prev_query_embed[img_i] for img_i, _, _ in origins],
                    dim=0,
                )

            (
                mask_logits_per_layer,
                class_logits_per_layer,
                procedure_logits_per_layer,
                crop_query_embed,
            ) = self(
                crops,
                time_indices=crop_time_indices,
                prev_query_embed=crop_prev_query_embed,
                return_query_embedding=True,
            )

            next_prev_query_embed = torch.zeros(
                bsz,
                crop_query_embed.shape[1],
                crop_query_embed.shape[2],
                device=crop_query_embed.device,
                dtype=crop_query_embed.dtype,
            )
            counts = torch.zeros(bsz, device=crop_query_embed.device, dtype=crop_query_embed.dtype)
            for crop_idx, (img_i, _, _) in enumerate(origins):
                next_prev_query_embed[img_i] += crop_query_embed[crop_idx]
                counts[img_i] += 1.0
            prev_query_embed = next_prev_query_embed / counts[:, None, None].clamp_min(1.0)

            self.log_procedure_accuracy(
                procedure_logits_per_layer,
                frame_targets,
                log_prefix,
                origins,
            )

            per_pixel_targets = self.to_per_pixel_targets_semantic(
                frame_targets,
                self.ignore_idx,
            )

            for i, (mask_logits, class_logits) in enumerate(
                list(zip(mask_logits_per_layer, class_logits_per_layer))
            ):
                mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
                crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
                logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

                self.update_metrics_semantic(logits, per_pixel_targets, i)

                if batch_idx == 0 and t == 0:
                    self.plot_semantic(
                        frame_imgs[0],
                        per_pixel_targets[0],
                        logits[0],
                        log_prefix,
                        i,
                        batch_idx,
                    )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
