"""
loss.py - Funzione di Perdita YOLO Anchor-Based per VGG16-BN + Neckless YOLO Head

Implementazione completa della YOLO Loss:
1. Allineamento ground-truth / ancore tramite IoU 2D.
2. Regressione Bounding Box:
   - Supporto a Smooth L1 (su tx, ty, tw, th) o CIoU (Complete IoU).
3. Perdita Objectness:
   - BCEWithLogits con separazione tra positive (`obj_weight`) e negative (`noobj_weight`).
   - Ignore threshold: le ancore senza target ma con IoU > threshold rispetto a un ground truth
     vengono escluse dalla penalizzazione no-object per evitare gradienti contraddittori.
4. Perdita Classificazione:
   - BCEWithLogits sulle classi con target one-hot per le sole ancore positive.
5. Robustezza numerica: gestione corretta di batch vuoti (0 box) e clipping dei logaritmi.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou_wh(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """
    Calcola l'IoU tra insiemi di dimensioni (w, h) assumendo che siano centrati nell'origine (0, 0).
    wh1: (N, 2)
    wh2: (M, 2)
    Ritorna: (N, M) matrice di IoU
    """
    wh1 = wh1.unsqueeze(1)  # (N, 1, 2)
    wh2 = wh2.unsqueeze(0)  # (1, M, 2)

    inter_w = torch.min(wh1[..., 0], wh2[..., 0])
    inter_h = torch.min(wh1[..., 1], wh2[..., 1])
    inter = inter_w * inter_h

    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]
    union = area1 + area2 - inter + 1e-16

    return inter / union


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calcola il Complete IoU (CIoU) tra box in coordinate [cx, cy, w, h] normalizzate.
    box1: (N, 4)
    box2: (N, 4)
    Ritorna: (N,) valori di CIoU in [-1, 1]
    """
    # Da (cx, cy, w, h) a (x1, y1, x2, y2)
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Intersezione
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Unione
    w1, h1 = box1[:, 2], box1[:, 3]
    w2, h2 = box2[:, 2], box2[:, 3]
    union_area = w1 * h1 + w2 * h2 - inter_area + 1e-16
    iou = inter_area / union_area

    # Minimo box contenitore enclosing
    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-16

    # Distanza tra i centri
    rho2 = (box1[:, 0] - box2[:, 0]) ** 2 + (box1[:, 1] - box2[:, 1]) ** 2

    # Fattore di consistenza dell'aspect ratio
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + 1e-16)) - torch.atan(w1 / (h1 + 1e-16)), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-16)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


class YOLOLoss(nn.Module):
    """
    Funzione di perdita YOLO per rilevatore VGG16-BN single-scale (stride 32).
    """

    def __init__(
        self,
        anchors: torch.Tensor,
        num_classes: int = 20,
        box_loss_weight: float = 5.0,
        obj_loss_weight: float = 1.0,
        noobj_loss_weight: float = 0.5,
        class_loss_weight: float = 1.0,
        ignore_threshold: float = 0.5,
        loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.register_buffer("anchors", anchors.clone().float())
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.box_loss_weight = box_loss_weight
        self.obj_loss_weight = obj_loss_weight
        self.noobj_loss_weight = noobj_loss_weight
        self.class_loss_weight = class_loss_weight
        self.ignore_threshold = ignore_threshold
        self.loss_type = loss_type.lower()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="sum", beta=1.0 / 9.0)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcola la perdita YOLO per il batch.

        Parametri:
            preds: (B, num_anchors * (5 + num_classes), S_h, S_w)
            targets: (N_targets, 6) con [batch_idx, class_id, cx, cy, w, h] normalizzati in [0, 1]

        Ritorna:
            total_loss: scalare con gradiente
            loss_components: dizionario con i valori numerici per logging
        """
        device = preds.device
        B, C, S_h, S_w = preds.shape
        A = self.num_anchors
        nc = self.num_classes

        # Reshape: (B, A, 5 + nc, S_h, S_w) -> (B, A, S_h, S_w, 5 + nc)
        preds = preds.view(B, A, 5 + nc, S_h, S_w).permute(0, 1, 3, 4, 2).contiguous()

        # Tensori target per coordinate, objectness e classi
        obj_mask = torch.zeros((B, A, S_h, S_w), dtype=torch.bool, device=device)
        noobj_mask = torch.ones((B, A, S_h, S_w), dtype=torch.bool, device=device)
        target_obj = torch.zeros((B, A, S_h, S_w), dtype=torch.float32, device=device)

        tx = torch.zeros((B, A, S_h, S_w), dtype=torch.float32, device=device)
        ty = torch.zeros((B, A, S_h, S_w), dtype=torch.float32, device=device)
        tw = torch.zeros((B, A, S_h, S_w), dtype=torch.float32, device=device)
        th = torch.zeros((B, A, S_h, S_w), dtype=torch.float32, device=device)
        target_cls = torch.zeros((B, A, S_h, S_w, nc), dtype=torch.float32, device=device)

        anchors = self.anchors.to(device)  # (A, 2) normalizzate in [0, 1]

        # Coordinate di griglia per la decodifica locale
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S_h, device=device),
            torch.arange(S_w, device=device),
            indexing="ij",
        )

        pred_boxes_all = None
        if targets.numel() > 0:
            # Calcolo predizioni decodificate per calcolo ignore mask
            pred_bx = (torch.sigmoid(preds[..., 0]) + grid_x.unsqueeze(0).unsqueeze(0)) / float(S_w)
            pred_by = (torch.sigmoid(preds[..., 1]) + grid_y.unsqueeze(0).unsqueeze(0)) / float(S_h)
            pred_bw = anchors[:, 0].view(1, A, 1, 1) * torch.exp(torch.clamp(preds[..., 2], max=10.0))
            pred_bh = anchors[:, 1].view(1, A, 1, 1) * torch.exp(torch.clamp(preds[..., 3], max=10.0))
            pred_boxes_all = torch.stack([pred_bx, pred_by, pred_bw, pred_bh], dim=-1)

        # Elaborazione ground-truth
        num_targets = targets.shape[0]
        pos_pred_boxes = []
        pos_gt_boxes = []

        for t_idx in range(num_targets):
            b = int(targets[t_idx, 0].item())
            cls_id = int(targets[t_idx, 1].item())
            gt_cx = targets[t_idx, 2]
            gt_cy = targets[t_idx, 3]
            gt_w = targets[t_idx, 4]
            gt_h = targets[t_idx, 5]

            # Ignora box degeneri
            if gt_w <= 0 or gt_h <= 0 or b >= B:
                continue

            # Calcolo cella di griglia
            gj = int(torch.clamp(gt_cx * S_w, 0, S_w - 1).item())
            gi = int(torch.clamp(gt_cy * S_h, 0, S_h - 1).item())

            # Calcola IoU con ciascuna delle ancore centrate nell'origine
            gt_wh = torch.tensor([[gt_w, gt_h]], device=device)
            anchor_ious = bbox_iou_wh(gt_wh, anchors).squeeze(0)  # (A,)
            best_a = int(torch.argmax(anchor_ious).item())

            # Assegnazione target all'ancora migliore
            obj_mask[b, best_a, gi, gj] = True
            noobj_mask[b, best_a, gi, gj] = False
            target_obj[b, best_a, gi, gj] = 1.0

            # Target coordinate locali della cella
            tx[b, best_a, gi, gj] = gt_cx * S_w - gj
            ty[b, best_a, gi, gj] = gt_cy * S_h - gi
            tw[b, best_a, gi, gj] = torch.log(gt_w / (anchors[best_a, 0] + 1e-16) + 1e-16)
            th[b, best_a, gi, gj] = torch.log(gt_h / (anchors[best_a, 1] + 1e-16) + 1e-16)

            # Target classificazione (one-hot)
            if 0 <= cls_id < nc:
                target_cls[b, best_a, gi, gj, cls_id] = 1.0

            if self.loss_type == "ciou" and pred_boxes_all is not None:
                pos_pred_boxes.append(pred_boxes_all[b, best_a, gi, gj])
                pos_gt_boxes.append(torch.tensor([gt_cx, gt_cy, gt_w, gt_h], device=device))

        # Maschera no-object: ignora celle con IoU > ignore_threshold con qualsiasi GT dell'immagine
        if pred_boxes_all is not None and self.ignore_threshold > 0:
            for b in range(B):
                b_targets = targets[targets[:, 0] == b]
                if b_targets.shape[0] == 0:
                    continue
                # Box predetti per l'immagine b: (A * S_h * S_w, 4)
                p_boxes = pred_boxes_all[b].view(-1, 4)
                gt_boxes = b_targets[:, 2:6]  # (M, 4)

                # Calcolo IoU box completo (intersezione / unione)
                # Calcola pairwise IoU
                p_x1 = p_boxes[:, 0] - p_boxes[:, 2] / 2
                p_y1 = p_boxes[:, 1] - p_boxes[:, 3] / 2
                p_x2 = p_boxes[:, 0] + p_boxes[:, 2] / 2
                p_y2 = p_boxes[:, 1] + p_boxes[:, 3] / 2

                g_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
                g_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
                g_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
                g_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

                ix1 = torch.max(p_x1.unsqueeze(1), g_x1.unsqueeze(0))
                iy1 = torch.max(p_y1.unsqueeze(1), g_y1.unsqueeze(0))
                ix2 = torch.min(p_x2.unsqueeze(1), g_x2.unsqueeze(0))
                iy2 = torch.min(p_y2.unsqueeze(1), g_y2.unsqueeze(0))

                inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
                union = p_boxes[:, 2:3] * p_boxes[:, 3:4] + (gt_boxes[:, 2] * gt_boxes[:, 3]).unsqueeze(0) - inter + 1e-16
                ious = inter / union  # (A * S_h * S_w, M)
                max_ious, _ = torch.max(ious, dim=1)
                ignore = (max_ious > self.ignore_threshold).view(A, S_h, S_w)
                noobj_mask[b][ignore] = False

        # Non penalizzare le ancore positive nel ramo noobj
        noobj_mask[obj_mask] = False

        # -------------------------------------------------------------
        # 1. Coordinate Loss
        # -------------------------------------------------------------
        num_pos = obj_mask.sum().item()
        if num_pos > 0:
            if self.loss_type == "ciou" and len(pos_pred_boxes) > 0:
                p_b = torch.stack(pos_pred_boxes, dim=0)
                g_b = torch.stack(pos_gt_boxes, dim=0)
                loss_box = (1.0 - bbox_ciou(p_b, g_b)).sum() / float(num_pos)
            else:
                # Regressione diretta con Smooth L1
                pred_x = torch.sigmoid(preds[..., 0][obj_mask])
                pred_y = torch.sigmoid(preds[..., 1][obj_mask])
                pred_w = preds[..., 2][obj_mask]
                pred_h = preds[..., 3][obj_mask]

                tgt_x = tx[obj_mask]
                tgt_y = ty[obj_mask]
                tgt_w = tw[obj_mask]
                tgt_h = th[obj_mask]

                loss_x = self.smooth_l1(pred_x, tgt_x)
                loss_y = self.smooth_l1(pred_y, tgt_y)
                loss_w = self.smooth_l1(pred_w, tgt_w)
                loss_h = self.smooth_l1(pred_h, tgt_h)
                loss_box = (loss_x + loss_y + loss_w + loss_h) / float(num_pos)

            # -------------------------------------------------------------
            # 2. Classification Loss (solo sulle ancore positive)
            # -------------------------------------------------------------
            pred_cls_pos = preds[..., 5:][obj_mask]  # (num_pos, nc)
            tgt_cls_pos = target_cls[obj_mask]       # (num_pos, nc)
            loss_cls = self.bce_loss(pred_cls_pos, tgt_cls_pos).sum() / float(num_pos)
        else:
            loss_box = torch.tensor(0.0, device=device)
            loss_cls = torch.tensor(0.0, device=device)

        # -------------------------------------------------------------
        # 3. Objectness Loss (Positive + Negative)
        # -------------------------------------------------------------
        pred_obj = preds[..., 4]
        bce_obj_all = self.bce_loss(pred_obj, target_obj)

        if num_pos > 0:
            loss_obj_pos = bce_obj_all[obj_mask].sum() / float(num_pos)
        else:
            loss_obj_pos = torch.tensor(0.0, device=device)

        num_noobj = noobj_mask.sum().item()
        if num_noobj > 0:
            loss_obj_neg = bce_obj_all[noobj_mask].sum() / float(num_noobj)
        else:
            loss_obj_neg = torch.tensor(0.0, device=device)

        loss_obj = self.obj_loss_weight * loss_obj_pos + self.noobj_loss_weight * loss_obj_neg

        # -------------------------------------------------------------
        # Perdita Totale Ponderata
        # -------------------------------------------------------------
        total_loss = (
            self.box_loss_weight * loss_box
            + loss_obj
            + self.class_loss_weight * loss_cls
        )

        loss_components = {
            "loss_total": float(total_loss.item()),
            "loss_box": float(loss_box.item()),
            "loss_obj": float(loss_obj.item()),
            "loss_obj_pos": float(loss_obj_pos.item()),
            "loss_obj_neg": float(loss_obj_neg.item()),
            "loss_cls": float(loss_cls.item()),
            "num_pos": num_pos,
        }

        return total_loss, loss_components

