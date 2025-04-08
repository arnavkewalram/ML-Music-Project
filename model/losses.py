"""
Loss Functions Module for Music Transcription System

This module defines custom loss functions for training the music transcription model.
It includes loss functions for onset detection, pitch detection, duration estimation,
tempo detection, time signature detection, and instrument classification.

Functions:
    onset_detection_loss: Loss function for onset detection
    pitch_detection_loss: Loss function for pitch detection
    duration_estimation_loss: Loss function for duration estimation
    tempo_detection_loss: Loss function for tempo detection
    time_signature_loss: Loss function for time signature detection
    instrument_classification_loss: Loss function for instrument classification
    velocity_estimation_loss: Loss function for velocity estimation
    combined_loss: Combined loss function for all tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def onset_detection_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 5.0
) -> torch.Tensor:
    """
    Loss function for onset detection.
    
    Args:
        predictions (torch.Tensor): Predicted onset probabilities of shape (batch_size, time_steps)
        targets (torch.Tensor): Target onset labels of shape (batch_size, time_steps)
        pos_weight (float): Weight for positive examples to handle class imbalance
        
    Returns:
        torch.Tensor: Onset detection loss
    """
    # Create weight tensor for positive examples
    weight = torch.ones_like(targets)
    weight[targets > 0] = pos_weight
    
    # Binary cross entropy loss with logits
    loss = F.binary_cross_entropy(predictions, targets, weight=weight)
    
    return loss


def pitch_detection_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    onset_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Loss function for pitch detection.
    
    Args:
        predictions (torch.Tensor): Predicted pitch probabilities of shape (batch_size, time_steps, num_pitches)
        targets (torch.Tensor): Target pitch labels of shape (batch_size, time_steps, num_pitches)
        onset_mask (torch.Tensor, optional): Mask for onset frames of shape (batch_size, time_steps)
        
    Returns:
        torch.Tensor: Pitch detection loss
    """
    # Binary cross entropy loss for multi-label classification
    loss = F.binary_cross_entropy(predictions, targets)
    
    # Apply onset mask if provided
    if onset_mask is not None:
        # Expand mask to match predictions shape
        mask = onset_mask.unsqueeze(-1).expand_as(predictions)
        
        # Apply mask to focus on frames with onsets
        loss = loss * mask
        
        # Normalize by sum of mask
        loss = loss.sum() / (mask.sum() + 1e-8)
    
    return loss


def duration_estimation_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    onset_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Loss function for duration estimation.
    
    Args:
        predictions (torch.Tensor): Predicted duration probabilities of shape (batch_size, time_steps, max_duration)
        targets (torch.Tensor): Target duration indices of shape (batch_size, time_steps)
        onset_mask (torch.Tensor, optional): Mask for onset frames of shape (batch_size, time_steps)
        
    Returns:
        torch.Tensor: Duration estimation loss
    """
    # Cross entropy loss for classification
    loss = F.cross_entropy(
        predictions.reshape(-1, predictions.size(-1)),
        targets.reshape(-1),
        reduction='none'
    ).reshape(targets.shape)
    
    # Apply onset mask if provided
    if onset_mask is not None:
        # Apply mask to focus on frames with onsets
        loss = loss * onset_mask
        
        # Normalize by sum of mask
        loss = loss.sum() / (onset_mask.sum() + 1e-8)
    else:
        # Take mean if no mask provided
        loss = loss.mean()
    
    return loss


def tempo_detection_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Loss function for tempo detection.
    
    Args:
        predictions (torch.Tensor): Predicted tempo of shape (batch_size, 1)
        targets (torch.Tensor): Target tempo of shape (batch_size, 1)
        
    Returns:
        torch.Tensor: Tempo detection loss
    """
    # Mean squared error loss
    loss = F.mse_loss(predictions, targets)
    
    return loss


def time_signature_loss(
    numerator_predictions: torch.Tensor,
    denominator_predictions: torch.Tensor,
    numerator_targets: torch.Tensor,
    denominator_targets: torch.Tensor
) -> torch.Tensor:
    """
    Loss function for time signature detection.
    
    Args:
        numerator_predictions (torch.Tensor): Predicted numerator probabilities of shape (batch_size, num_numerators)
        denominator_predictions (torch.Tensor): Predicted denominator probabilities of shape (batch_size, num_denominators)
        numerator_targets (torch.Tensor): Target numerator indices of shape (batch_size,)
        denominator_targets (torch.Tensor): Target denominator indices of shape (batch_size,)
        
    Returns:
        torch.Tensor: Time signature detection loss
    """
    # Cross entropy loss for numerator
    numerator_loss = F.cross_entropy(numerator_predictions, numerator_targets)
    
    # Cross entropy loss for denominator
    denominator_loss = F.cross_entropy(denominator_predictions, denominator_targets)
    
    # Combine losses
    loss = numerator_loss + denominator_loss
    
    return loss


def instrument_classification_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Loss function for instrument classification.
    
    Args:
        predictions (torch.Tensor): Predicted instrument probabilities of shape (batch_size, num_instruments)
        targets (torch.Tensor): Target instrument labels of shape (batch_size, num_instruments)
        
    Returns:
        torch.Tensor: Instrument classification loss
    """
    # Binary cross entropy loss for multi-label classification
    loss = F.binary_cross_entropy(predictions, targets)
    
    return loss


def velocity_estimation_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    onset_mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Loss function for velocity estimation.
    
    Args:
        predictions (torch.Tensor): Predicted velocities of shape (batch_size, time_steps)
        targets (torch.Tensor): Target velocities of shape (batch_size, time_steps)
        onset_mask (torch.Tensor, optional): Mask for onset frames of shape (batch_size, time_steps)
        
    Returns:
        torch.Tensor: Velocity estimation loss
    """
    # Mean squared error loss
    loss = F.mse_loss(predictions, targets, reduction='none')
    
    # Apply onset mask if provided
    if onset_mask is not None:
        # Apply mask to focus on frames with onsets
        loss = loss * onset_mask
        
        # Normalize by sum of mask
        loss = loss.sum() / (onset_mask.sum() + 1e-8)
    else:
        # Take mean if no mask provided
        loss = loss.mean()
    
    return loss


def combined_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, float] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined loss function for all tasks.
    
    Args:
        predictions (Dict[str, torch.Tensor]): Dictionary of predictions from all modules
        targets (Dict[str, torch.Tensor]): Dictionary of targets for all modules
        weights (Dict[str, float], optional): Dictionary of weights for each loss component
        
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and dictionary of individual losses
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'onset': 1.0,
            'pitch': 1.0,
            'duration': 1.0,
            'tempo': 0.5,
            'time_signature': 0.5,
            'instrument': 0.5,
            'velocity': 0.5
        }
    
    # Calculate individual losses
    losses = {}
    
    # Onset detection loss
    if 'onset_probs' in predictions and 'onset_targets' in targets:
        losses['onset'] = onset_detection_loss(
            predictions['onset_probs'],
            targets['onset_targets']
        )
    
    # Pitch detection loss
    if 'pitch_probs' in predictions and 'pitch_targets' in targets:
        onset_mask = targets.get('onset_targets', None)
        losses['pitch'] = pitch_detection_loss(
            predictions['pitch_probs'],
            targets['pitch_targets'],
            onset_mask
        )
    
    # Duration estimation loss
    if 'duration_probs' in predictions and 'duration_targets' in targets:
        onset_mask = targets.get('onset_targets', None)
        losses['duration'] = duration_estimation_loss(
            predictions['duration_probs'],
            targets['duration_targets'],
            onset_mask
        )
    
    # Tempo detection loss
    if 'tempo' in predictions and 'tempo_targets' in targets:
        losses['tempo'] = tempo_detection_loss(
            predictions['tempo'],
            targets['tempo_targets']
        )
    
    # Time signature loss
    if ('numerator_probs' in predictions and 'denominator_probs' in predictions and
        'numerator_targets' in targets and 'denominator_targets' in targets):
        losses['time_signature'] = time_signature_loss(
            predictions['numerator_probs'],
            predictions['denominator_probs'],
            targets['numerator_targets'],
            targets['denominator_targets']
        )
    
    # Instrument classification loss
    if 'instrument_probs' in predictions and 'instrument_targets' in targets:
        losses['instrument'] = instrument_classification_loss(
            predictions['instrument_probs'],
            targets['instrument_targets']
        )
    
    # Velocity estimation loss
    if 'velocity' in predictions and 'velocity_targets' in targets:
        onset_mask = targets.get('onset_targets', None)
        losses['velocity'] = velocity_estimation_loss(
            predictions['velocity'],
            targets['velocity_targets'],
            onset_mask
        )
    
    # Calculate total loss
    total_loss = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    
    return total_loss, losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Attributes:
        alpha (float): Weighting factor for the rare class
        gamma (float): Focusing parameter
        reduction (str): Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize the FocalLoss with specified parameters.
        
        Args:
            alpha (float): Weighting factor for the rare class
            gamma (float): Focusing parameter
            reduction (str): Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FocalLoss.
        
        Args:
            inputs (torch.Tensor): Predicted probabilities
            targets (torch.Tensor): Target labels
            
        Returns:
            torch.Tensor: Focal loss
        """
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal weights
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
