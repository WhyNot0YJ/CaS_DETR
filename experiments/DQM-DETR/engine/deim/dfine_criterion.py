"""
D-FINE criterion extensions for DQM-DETR.
"""

import torch
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, paired_box_iou_xyxy
from .deim_criterion import DEIMCriterion
from ..core import register


@register()
class DQMDFINECriterion(DEIMCriterion):
    def __init__(self, *args, qmqc_weight=0.25, qmqc_eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        if 'mal' in self.losses:
            raise ValueError('DQMDFINECriterion uses D-FINE losses and does not support mal')
        self.qmqc_weight = qmqc_weight
        self.qmqc_eps = qmqc_eps

    def _prefix_losses(self, losses, prefix):
        return {f'{prefix}_{k}': v for k, v in losses.items()}

    def _loss_qmqc(self, clean_outputs, degraded_outputs, targets):
        if 'query_embed' not in clean_outputs or 'query_embed' not in degraded_outputs:
            return clean_outputs['pred_logits'].sum() * 0.0

        clean_indices = self.matcher({k: v for k, v in clean_outputs.items() if 'aux' not in k}, targets)['indices']
        degraded_indices = self.matcher({k: v for k, v in degraded_outputs.items() if 'aux' not in k}, targets)['indices']

        losses = []
        weights = []
        for batch_idx, ((clean_src, clean_tgt), (deg_src, deg_tgt), target) in enumerate(zip(clean_indices, degraded_indices, targets)):
            if clean_src.numel() == 0 or deg_src.numel() == 0:
                continue

            deg_by_gt = {int(t.item()): int(s.item()) for s, t in zip(deg_src, deg_tgt)}
            clean_pairs = [(int(s.item()), int(t.item())) for s, t in zip(clean_src, clean_tgt) if int(t.item()) in deg_by_gt]
            if len(clean_pairs) == 0:
                continue

            clean_query_idx = torch.tensor([p[0] for p in clean_pairs], device=clean_src.device, dtype=torch.long)
            gt_idx = torch.tensor([p[1] for p in clean_pairs], device=clean_src.device, dtype=torch.long)
            deg_query_idx = torch.tensor([deg_by_gt[p[1]] for p in clean_pairs], device=clean_src.device, dtype=torch.long)

            clean_query = clean_outputs['query_embed'][batch_idx, clean_query_idx].detach()
            degraded_query = degraded_outputs['query_embed'][batch_idx, deg_query_idx]
            query_loss = 1.0 - F.cosine_similarity(degraded_query, clean_query, dim=-1)

            labels = target['labels'][gt_idx]
            clean_score = clean_outputs['pred_logits'][batch_idx, clean_query_idx].sigmoid().gather(1, labels[:, None]).squeeze(1)
            clean_boxes = clean_outputs['pred_boxes'][batch_idx, clean_query_idx]
            target_boxes = target['boxes'][gt_idx]
            clean_iou = paired_box_iou_xyxy(
                box_cxcywh_to_xyxy(clean_boxes.detach()),
                box_cxcywh_to_xyxy(target_boxes)
            )
            quality = (clean_score.detach() * clean_iou.detach()).clamp(min=0.0)

            losses.append(query_loss)
            weights.append(quality)

        if len(losses) == 0:
            return clean_outputs['pred_logits'].sum() * 0.0

        losses = torch.cat(losses)
        weights = torch.cat(weights)
        return (losses * weights).sum() / weights.sum().clamp(min=self.qmqc_eps)

    def forward(self, outputs, targets, **kwargs):
        if not isinstance(outputs, dict) or 'clean' not in outputs or 'degraded' not in outputs:
            return super().forward(outputs, targets, **kwargs)

        clean_losses = self._prefix_losses(super().forward(outputs['clean'], targets, **kwargs), 'clean')
        degraded_losses = self._prefix_losses(super().forward(outputs['degraded'], targets, **kwargs), 'degraded')
        loss_qmqc = self._loss_qmqc(outputs['clean'], outputs['degraded'], targets) * self.qmqc_weight

        losses = {}
        losses.update(clean_losses)
        losses.update(degraded_losses)
        losses['loss_qmqc'] = loss_qmqc
        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
