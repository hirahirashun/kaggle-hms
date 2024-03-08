import numpy as np
import torch
import torch.nn as nn


def get_loss_fn(model_name, pred_confidence=False):
    if (model_name=="HMSSpecPararellModel") or (model_name=="HMSSpecEEGPararellModel") or (model_name=="HMSSpecEEGThreePararellModel"):
        loss_fn = KLDWithContrastiveLoss(pred_confidence=pred_confidence)
    else:
        loss_fn = KLDivLossWithLogits()

    return loss_fn

class KLDivLossWithLogits(nn.KLDivLoss):

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t, level_t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss


class KLDivLossWithLogitsForVal(nn.KLDivLoss):
    
    def __init__(self):
        """"""
        super().__init__(reduction="batchmean")
        self.log_prob_list  = []
        self.label_list = []

    def forward(self, y, t):
        y = nn.functional.log_softmax(y, dim=1)
        self.log_prob_list.append(y.detach().cpu().to(torch.float32).numpy())
        self.label_list.append(t.detach().cpu().to(torch.float32).numpy())
        
    def compute(self):
        log_prob = np.concatenate(self.log_prob_list, axis=0)
        label = np.concatenate(self.label_list, axis=0)
        final_metric = super().forward(
            torch.from_numpy(log_prob),
            torch.from_numpy(label)
        ).item()
        self.log_prob_list = []
        self.label_list = []
        
        return final_metric
    

class KLDWithContrastiveLoss(nn.Module):
    def __init__(self, pred_confidence):
        super().__init__()
        self.classification_loss = KLDivLossWithLogits()
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.pred_confidence = pred_confidence

    def forward(self, y, t, level_t):
        cls_loss = self.classification_loss(y['output'], t, level_t)
        cls_loss_original = self.classification_loss(y['original_output'], t, level_t)
        cls_loss_eeg = self.classification_loss(y['eeg_output'], t, level_t)

        # Calculate contrastive loss 
        
        embedding_original = torch.nn.functional.normalize(y["original_feat"], p=2, dim=1)
        embedding_eeg = torch.nn.functional.normalize(y["eeg_feat"], p=2, dim=1)

        
        contrastive_target = torch.ones(embedding_original.size(0)).to(y['output'].device)  # Assuming all pairs are similar
        contrastive_loss = self.contrastive_loss(embedding_original, embedding_eeg, contrastive_target)

        total_loss = cls_loss + 0.5*cls_loss_original + 0.5*cls_loss_eeg + 0.5*contrastive_loss # Aux losses

        if self.pred_confidence:
            cls_loss_weighted = self.classification_loss(y['weighted_output'], t, level_t)
            total_loss += 0.5*cls_loss_weighted

        return {
                'loss': total_loss, 
                'cls_loss': cls_loss, 
                "cls_loss_original": cls_loss_original, 
                "cls_loss_eeg": cls_loss_eeg, 
                'contrastive_loss': contrastive_loss
                }
    
class KLDWithThreeModalContrastiveLoss(nn.Module):
    def __init__(self, pred_confidence=False):
        super().__init__()
        self.classification_loss = KLDivLossWithLogits()
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.pred_confidence = pred_confidence

    def forward(self, y, t, level_t):
        cls_loss = self.classification_loss(y['output'], t, level_t)
        cls_loss_original_spec = self.classification_loss(y['original_spec_output'], t, level_t)
        cls_loss_eeg_spec = self.classification_loss(y['eeg_spec_output'], t, level_t)
        cls_loss_eeg = self.classification_loss(y['eeg_output'], t, level_t)

        # Calculate contrastive loss 
        
        embedding_original_spec = torch.nn.functional.normalize(y["original_spec_feat"], p=2, dim=1)
        embedding_eeg_spec = torch.nn.functional.normalize(y["eeg_spec_feat"], p=2, dim=1)
        embedding_eeg = torch.nn.functional.normalize(y["eeg_feat"], p=2, dim=1)

        
        contrastive_target = torch.ones(embedding_original_spec.size(0)).to(y['output'].device)  # Assuming all pairs are similar
        contrastive_loss = self.contrastive_loss(embedding_original_spec, embedding_eeg, contrastive_target)
        contrastive_loss += self.contrastive_loss(embedding_eeg_spec, embedding_eeg, contrastive_target)
        contrastive_loss += self.contrastive_loss(embedding_eeg_spec, embedding_original_spec, contrastive_target)
        contrastive_loss /= 3.0

        total_loss = cls_loss + 0.5*cls_loss_original_spec + 0.5*cls_loss_eeg_spec + 0.5*cls_loss_eeg + 0.5*contrastive_loss # Aux losses

        

        return {
                'loss': total_loss, 
                'cls_loss': cls_loss, 
                "cls_loss_original_spec": cls_loss_original_spec, 
                "cls_loss_eeg_spec": cls_loss_eeg_spec, 
                "cls_loss_eeg": cls_loss_eeg, 
                'contrastive_loss': contrastive_loss
                }
    
