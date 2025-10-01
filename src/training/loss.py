import kornia

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEDiceLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1e-5):
        """
        Combined Cross Entropy (CE) and Dice Loss.

        Args:
            weight_dice (float): Weight for the Dice loss component.
            weight_ce (float): Weight for the Cross-Entropy loss component.
            smooth (float): Smoothing factor to avoid division by zero in Dice loss.
        """
        super(CEDiceLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()  # Cross Entropy for multi-class segmentation
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.smooth = smooth

    def dice_loss(self, probs, target):
        """
        Compute Dice Loss.

        Args:
            probs (torch.Tensor): Softmaxed predictions (probabilities), shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """

        # Convert target to one-hot encoding
        num_classes = probs.shape[1]  # Number of classes (C)
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # Shape: (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)

        # Compute Dice loss
        intersection = (probs * target_one_hot).sum(dim=(2, 3))  # Sum over spatial dims
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Return Dice loss

    def forward(self, logits, target):
        """
        Compute the combined CE + Dice loss.

        Args:
            logits (torch.Tensor): Raw logits from the model, shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """
        # Compute Cross-Entropy Loss
        ce_loss = self.ce(logits, target.long())  # Cross-Entropy expects raw logits

        # Convert logits to softmax probabilities for Dice loss
        probs = torch.softmax(logits, dim=1)  # Convert to probabilities

        # Compute Dice Loss
        dice_loss = self.dice_loss(probs, target)

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


class DiceBoundaryLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_boundary=0.5, smooth=1e-5):
        """
        Combined Dice and Boundary Loss.

        Args:
            weight_dice (float): Weight for the Dice loss component.
            weight_boundary (float): Weight for the Boundary loss component.
            smooth (float): Smoothing factor to avoid division by zero in Dice loss.
        """
        super(DiceBoundaryLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_boundary = weight_boundary
        self.smooth = smooth

    def dice_loss(self, probs, target):
        """
        Compute Dice Loss.

        Args:
            probs (torch.Tensor): Softmaxed predictions (probabilities), shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """
        num_classes = probs.shape[1]  # Number of classes (C)
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # Shape: (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # Shape: (B, C, H, W)

        # Compute Dice loss
        intersection = (probs * target_one_hot).sum(dim=(2, 3))  # Sum over spatial dims
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Return Dice loss

    def boundary_loss(self, probs, target):
        """
        Compute Boundary Loss using distance transform.

        Args:
            probs (torch.Tensor): Softmaxed predictions (probabilities), shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """
        num_classes = probs.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Compute distance transform of the target mask
        target_boundary = kornia.filters.sobel(target_one_hot).sum(dim=1, keepdim=True)  # Edge detection
        boundary_weight = torch.exp(-target_boundary)  # Exponential weighting to emphasize boundary

        # Boundary loss = weighted sum of absolute differences
        boundary_loss = (boundary_weight * torch.abs(probs - target_one_hot)).mean()
        return boundary_loss

    def forward(self, logits, target):
        """
        Compute the combined Dice + Boundary loss.

        Args:
            logits (torch.Tensor): Raw logits from the model, shape (B, C, H, W)
            target (torch.Tensor): Ground truth class indices, shape (B, H, W)
        """
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities

        dice_loss = self.dice_loss(probs, target)
        boundary_loss = self.boundary_loss(probs, target)

        return self.weight_dice * dice_loss + self.weight_boundary * boundary_loss



class DiceFocalLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_focal=0.5, alpha=0.25, gamma=2.0, smooth=1e-5):
        """
        Combined Dice and Focal Loss for multi-class segmentation.

        Args:
            weight_dice (float): Weight for the Dice loss component.
            weight_focal (float): Weight for the Focal loss component.
            alpha (float): Focal Loss weighting factor for class balance (set to None if not needed).
            gamma (float): Focusing parameter for Focal Loss (higher = focuses more on hard examples).
            smooth (float): Smoothing factor to avoid division by zero in Dice loss.
        """
        super(DiceFocalLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def dice_loss(self, probs, target):
        """
        Compute Dice Loss.

        Args:
            probs (torch.Tensor): Softmaxed predictions (probabilities), shape (B, C, H, W).
            target (torch.Tensor): Ground truth class indices, shape (B, H, W).
        """

        # Convert target to one-hot encoding
        num_classes = probs.shape[1]  # Number of classes (C)
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute Dice loss
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Return Dice loss

    def focal_loss(self, logits, target):
        """
        Compute Focal Loss.

        Args:
            logits (torch.Tensor): Raw logits from the model, shape (B, C, H, W).
            target (torch.Tensor): Ground truth class indices, shape (B, H, W).
        """
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        probs = probs.clamp(min=1e-5, max=1.0)  # Avoid log(0) issues

        # Compute Focal Loss
        ce_loss = -target_one_hot * torch.log(probs)  # Cross-entropy loss
        if self.alpha is not None:
            alpha_factor = self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot)
            ce_loss *= alpha_factor  # Apply alpha weighting
        
        focal_weight = (1 - probs) ** self.gamma  # Compute focusing factor
        focal_loss = focal_weight * ce_loss  # Apply focal weight

        return focal_loss.mean()  # Return mean focal loss

    def forward(self, logits, target):
        """
        Compute the combined Dice + Focal Loss.

        Args:
            logits (torch.Tensor): Raw logits from the model, shape (B, C, H, W).
            target (torch.Tensor): Ground truth class indices, shape (B, H, W).
        """
        # Convert logits to softmax probabilities for Dice loss
        probs = torch.softmax(logits, dim=1)

        # Compute Dice and Focal Loss
        dice_loss = self.dice_loss(probs, target)
        focal_loss = self.focal_loss(logits, target)

        return self.weight_dice * dice_loss + self.weight_focal * focal_loss


class CEDiceLossWeighted(nn.Module):
    def __init__(self, weight_dice=1.0, weight_ce=1.0, smooth=1e-5, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.smooth = smooth
        self.class_weights = class_weights

    def dice_loss(self, probs, target):
        num_classes = probs.shape[1]
        one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum((2, 3))
        union = probs.sum((2, 3)) + one_hot.sum((2, 3))
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)

        if self.class_weights is not None:
            weights = self.class_weights.to(dice_per_class.device)
            weights = weights / weights.sum()
            weighted_dice = (1 - dice_per_class) * weights
            return weighted_dice.mean()
        else:
            return 1 - dice_per_class.mean()

    def forward(self, logits, target):
        ce = self.ce(logits, target.long())
        probs = torch.softmax(logits, dim=1)
        dice = self.dice_loss(probs, target)
        return self.weight_ce * ce + self.weight_dice * dice