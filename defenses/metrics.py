import torch
import torch.nn.functional as F
import numpy as np

def evaluate_model(model, test_loader, device='cpu'):
    """
    Fast vectorized evaluation for accuracy, Demographic Parity gap, and Equalized Odds gap.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_sensitives = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                x, y, sens = batch[0], batch[1], batch[2]
            else:
                x, y, sens = batch['x'], batch['y'], batch['sensitive']
                
            out = model(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            preds = torch.argmax(out, dim=1)
            all_preds.append(preds)
            all_targets.append(y)
            all_sensitives.append(sens)
            
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    sens = torch.cat(all_sensitives)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = float((preds == targets).float().mean().item())
    
    m_mask = (sens == 1)
    f_mask = (sens == 0)
    
    p_m = float((preds[m_mask] == 1).float().mean().item()) if m_mask.sum() > 0 else 0.0
    p_f = float((preds[f_mask] == 1).float().mean().item()) if f_mask.sum() > 0 else 0.0
    dp_gap = abs(p_m - p_f)
    
    # FPR & TPR differences
    y0_m = (targets == 0) & (sens == 1)
    y0_f = (targets == 0) & (sens == 0)
    fpr_m = float((preds[y0_m] == 1).float().mean().item()) if y0_m.sum() > 0 else 0.0
    fpr_f = float((preds[y0_f] == 1).float().mean().item()) if y0_f.sum() > 0 else 0.0
    
    y1_m = (targets == 1) & (sens == 1)
    y1_f = (targets == 1) & (sens == 0)
    tpr_m = float((preds[y1_m] == 1).float().mean().item()) if y1_m.sum() > 0 else 0.0
    tpr_f = float((preds[y1_f] == 1).float().mean().item()) if y1_f.sum() > 0 else 0.0
    
    eo_gap = max(abs(fpr_m - fpr_f), abs(tpr_m - tpr_f))
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'dp_gap': dp_gap,
        'eo_gap': eo_gap,
        'p_pred1_male': p_m,
        'p_pred1_female': p_f
    }
