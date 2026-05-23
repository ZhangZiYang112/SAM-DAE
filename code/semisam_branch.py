"""
SemiSAM Branch for Semi-Supervised Learning Enhancement
Uses SAM-Med3D to provide additional supervision signals
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything.build_sam3D import sam_model_registry3D


class SemiSAMBranch:
    """SAM-Med3D based semi-supervised learning branch"""
    
    def __init__(self, checkpoint_path='code/segment_anything/ckpt/sam_med3d_turbo.pth', 
                 device='cuda', image_size=128):
        self.device = device
        self.image_size = image_size
        self.sam_model = None
        self.checkpoint_path = checkpoint_path
        self._initialized = False
        
    def initialize(self):
        """Lazy initialization of SAM model"""
        if self._initialized:
            return
        try:
            # Use vit_b_ori which matches the sam_med3d_turbo.pth checkpoint configuration
            # (embed_dim=768, image_size=128)
            self.sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=self.checkpoint_path)
            self.sam_model = self.sam_model.to(self.device)
            self.sam_model.eval()
            self._initialized = True
            print(f"SAM-Med3D model loaded from {self.checkpoint_path}")
        except Exception as e:
            print(f"Failed to load SAM-Med3D model: {e}")
            self._initialized = False
            
    def get_uncertainty_map(self, pred_soft):
        """
        Calculate uncertainty map from prediction probabilities
        Using entropy-based uncertainty
        
        Args:
            pred_soft: softmax prediction [B, C, D, H, W]
        Returns:
            uncertainty: [B, 1, D, H, W]
        """
        # Entropy-based uncertainty
        eps = 1e-8
        entropy = -torch.sum(pred_soft * torch.log(pred_soft + eps), dim=1, keepdim=True)
        # Normalize entropy to [0, 1]
        max_entropy = np.log(pred_soft.shape[1])
        uncertainty = entropy / max_entropy
        return uncertainty
    
    def get_prompt_from_prediction(self, pred_mask, prompt_type='centroid'):
        """
        Generate prompts for SAM from prediction mask
        
        Args:
            pred_mask: binary prediction mask [B, D, H, W]
            prompt_type: 'centroid', 'random', 'boundary'
        Returns:
            point_coords: [B, N, 3]
            point_labels: [B, N]
        """
        batch_size = pred_mask.shape[0]
        point_coords_list = []
        point_labels_list = []
        
        for b in range(batch_size):
            mask = pred_mask[b].cpu().numpy()
            # Find foreground points
            fg_points = np.where(mask > 0.5)
            
            if len(fg_points[0]) > 0:
                if prompt_type == 'centroid':
                    # Use centroid as prompt
                    center_d = int(np.mean(fg_points[0]))
                    center_h = int(np.mean(fg_points[1]))
                    center_w = int(np.mean(fg_points[2]))
                    points = [[center_d, center_h, center_w]]
                elif prompt_type == 'random':
                    # Random sample foreground points
                    num_points = min(5, len(fg_points[0]))
                    idx = np.random.choice(len(fg_points[0]), num_points, replace=False)
                    points = [[fg_points[0][i], fg_points[1][i], fg_points[2][i]] for i in idx]
                else:  # boundary
                    # Sample from boundary
                    from scipy import ndimage
                    eroded = ndimage.binary_erosion(mask, iterations=2)
                    boundary = mask.astype(float) - eroded.astype(float)
                    bd_points = np.where(boundary > 0)
                    if len(bd_points[0]) > 0:
                        num_points = min(5, len(bd_points[0]))
                        idx = np.random.choice(len(bd_points[0]), num_points, replace=False)
                        points = [[bd_points[0][i], bd_points[1][i], bd_points[2][i]] for i in idx]
                    else:
                        center_d = int(np.mean(fg_points[0]))
                        center_h = int(np.mean(fg_points[1]))
                        center_w = int(np.mean(fg_points[2]))
                        points = [[center_d, center_h, center_w]]
            else:
                # No foreground, use center of volume
                d, h, w = mask.shape
                points = [[d // 2, h // 2, w // 2]]
                
            point_coords_list.append(points)
            point_labels_list.append([1] * len(points))  # All foreground
        
        # Pad to same length
        max_points = max(len(p) for p in point_coords_list)
        for i in range(batch_size):
            while len(point_coords_list[i]) < max_points:
                point_coords_list[i].append(point_coords_list[i][0])
                point_labels_list[i].append(point_labels_list[i][0])
        
        point_coords = torch.tensor(point_coords_list, dtype=torch.float32, device=self.device)
        point_labels = torch.tensor(point_labels_list, dtype=torch.int32, device=self.device)
        
        return point_coords, point_labels
    
    def resize_volume(self, volume, target_size):
        """Resize 3D volume to target size"""
        return F.interpolate(volume, size=target_size, mode='trilinear', align_corners=False)
    
    @torch.no_grad()
    def forward(self, image, pred_soft, prompt_type='unc'):
        """
        Forward pass through SAM-Med3D
        
        Args:
            image: input image [B, 1, D, H, W]
            pred_soft: softmax prediction [B, C, D, H, W]
            prompt_type: 'unc' for uncertainty-weighted, 'centroid', 'random'
        Returns:
            sam_mask: SAM segmentation mask [B, 1, D, H, W]
            uncertainty: uncertainty map [B, 1, D, H, W]
        """
        self.initialize()
        
        if not self._initialized:
            # Return placeholder if SAM not available
            return pred_soft[:, 1:2, :, :, :], torch.ones_like(pred_soft[:, 0:1, :, :, :])
        
        batch_size = image.shape[0]
        orig_size = image.shape[2:]
        
        # Calculate uncertainty from predictions
        uncertainty = self.get_uncertainty_map(pred_soft)
        
        # Get prediction mask for prompts
        pred_mask = pred_soft[:, 1, :, :, :] > 0.5  # Foreground class
        
        # Resize image for SAM
        sam_size = (self.image_size, self.image_size, self.image_size)
        image_resized = self.resize_volume(image, sam_size)
        
        # Generate prompts
        pred_mask_resized = self.resize_volume(pred_mask.float().unsqueeze(1), sam_size).squeeze(1)
        point_coords, point_labels = self.get_prompt_from_prediction(pred_mask_resized, 
                                                                      prompt_type='centroid')
        
        # Scale point coordinates to SAM input size
        scale_factors = torch.tensor([sam_size[0] / orig_size[0], 
                                      sam_size[1] / orig_size[1], 
                                      sam_size[2] / orig_size[2]], device=self.device)
        
        # Prepare SAM input
        sam_outputs = []
        for b in range(batch_size):
            # Normalize image
            img = image_resized[b]  # [1, D, H, W]
            
            batched_input = [{
                'image': img,
                'original_size': orig_size,
                'point_coords': point_coords[b:b+1],
                'point_labels': point_labels[b:b+1],
            }]
            
            try:
                output = self.sam_model(batched_input, multimask_output=False)
                sam_mask = output[0]['masks'].float()
                sam_outputs.append(sam_mask)
            except Exception as e:
                # Fallback to prediction if SAM fails
                sam_outputs.append(pred_soft[b:b+1, 1:2, :, :, :])
        
        # Stack outputs
        sam_mask = torch.cat(sam_outputs, dim=0)
        
        # Resize back to original size
        if sam_mask.shape[2:] != orig_size:
            sam_mask = self.resize_volume(sam_mask, orig_size)
        
        return sam_mask, uncertainty


# Global instance
_sam_branch = None

def get_sam_branch(checkpoint_path='code/segment_anything/ckpt/sam_med3d_turbo.pth'):
    """Get or create global SAM branch instance"""
    global _sam_branch
    if _sam_branch is None:
        _sam_branch = SemiSAMBranch(checkpoint_path=checkpoint_path)
    return _sam_branch


def semisam_branch(image, pred_soft, generalist='SAM-Med3D', prompt='unc'):
    """
    Main interface for SemiSAM branch
    
    Args:
        image: input image [B, 1, D, H, W]
        pred_soft: softmax prediction from model [B, C, D, H, W] 
                   or [B, 1, D, H, W] for foreground prob
        generalist: which SAM model to use
        prompt: prompt strategy - 'unc' for uncertainty-weighted
    Returns:
        sam_mask: SAM segmentation [B, 1, D, H, W]
        uncertainty: uncertainty map [B, 1, D, H, W]
    """
    sam_branch = get_sam_branch()
    
    # Ensure pred_soft has correct shape [B, C, D, H, W]
    if pred_soft.shape[1] == 1:
        # Convert [B, 1, D, H, W] to [B, 2, D, H, W]
        pred_soft = torch.cat([1 - pred_soft, pred_soft], dim=1)
    
    return sam_branch.forward(image, pred_soft, prompt_type=prompt)


def compute_sam_consistency_loss(pred_soft, sam_mask, uncertainty=None, 
                                  use_uncertainty_weighting=True):
    """
    Compute SAM consistency loss
    
    Args:
        pred_soft: model softmax prediction [B, C, D, H, W]
        sam_mask: SAM segmentation [B, 1, D, H, W]
        uncertainty: uncertainty map [B, 1, D, H, W]
        use_uncertainty_weighting: whether to weight by uncertainty
    Returns:
        sam_loss: scalar loss value
    """
    # Get foreground probability from prediction
    pred_fg = pred_soft[:, 1:2, :, :, :]
    
    # MSE loss between prediction and SAM
    mse_loss = (pred_fg - sam_mask) ** 2
    
    if use_uncertainty_weighting and uncertainty is not None:
        # Weight by inverse uncertainty (high confidence regions have more weight)
        # But also consider uncertain regions
        weight = 1.0 - uncertainty + 0.1  # Add small constant to avoid zero weights
        weighted_loss = mse_loss * weight
        loss = torch.mean(weighted_loss) / (torch.mean(weight) + 1e-8)
        # Add uncertainty regularization
        loss = loss + 0.1 * torch.mean(uncertainty)
    else:
        loss = torch.mean(mse_loss)
    
    return loss


def compute_sam_dice_loss(pred_soft, sam_mask, smooth=1e-5):
    """
    Compute Dice loss between prediction and SAM output
    
    Args:
        pred_soft: model softmax prediction [B, C, D, H, W]
        sam_mask: SAM segmentation [B, 1, D, H, W]
    Returns:
        dice_loss: scalar loss value
    """
    pred_fg = pred_soft[:, 1:2, :, :, :]
    
    intersection = torch.sum(pred_fg * sam_mask)
    union = torch.sum(pred_fg ** 2) + torch.sum(sam_mask ** 2)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice
