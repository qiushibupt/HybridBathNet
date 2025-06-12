import torch

def masked_mse_loss(
    pred_depth: torch.Tensor, 
    target_depth: torch.Tensor, 
    mask: torch.Tensor, 
    pred_wdi: torch.Tensor, 
    target_wdi: torch.Tensor,
    depth_weight: float = 0.9,
    wdi_weight: float = 0.1
) -> torch.Tensor:
    
    depth_loss = torch.square(pred_depth - target_depth) * mask
    depth_mse = depth_loss.sum() / mask.sum()
    
    wdi_loss = torch.square(pred_wdi - target_wdi) * mask
    wdi_mse = wdi_loss.sum() / mask.sum()

    return depth_weight * depth_mse + wdi_weight * wdi_mse 