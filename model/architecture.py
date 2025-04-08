"""
Model Architecture Module for Music Transcription System

This module defines the custom deep learning model architecture for music transcription.
It includes neural network components for onset detection, pitch detection, duration
estimation, tempo detection, time signature detection, and instrument classification.

Classes:
    OnsetDetectionModule: Detects note onsets from audio features
    PitchDetectionModule: Detects pitch of notes from audio features
    DurationEstimationModule: Estimates duration of notes
    TempoDetectionModule: Detects tempo from audio features
    TimeSignatureModule: Detects time signature from audio features
    InstrumentClassificationModule: Classifies instruments in audio
    MusicTranscriptionModel: Complete model combining all modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation.
    
    Attributes:
        conv (nn.Conv2d): Convolutional layer
        bn (nn.BatchNorm2d): Batch normalization layer
        activation (nn.Module): Activation function
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.ReLU(),
        use_bn: bool = True
    ):
        """
        Initialize the ConvBlock with specified parameters.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel
            stride (Union[int, Tuple[int, int]]): Stride of the convolution
            padding (Union[int, Tuple[int, int]]): Zero-padding added to both sides of the input
            dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements
            groups (int): Number of blocked connections from input to output channels
            bias (bool): If True, adds a learnable bias to the output
            activation (nn.Module): Activation function
            use_bn (bool): Whether to use batch normalization
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = self.conv(x)
        
        if self.use_bn:
            x = self.bn(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Attributes:
        conv1 (ConvBlock): First convolutional block
        conv2 (ConvBlock): Second convolutional block
        downsample (nn.Module): Optional downsampling layer for skip connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize the ResidualBlock with specified parameters.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (Union[int, Tuple[int, int]]): Stride of the convolution
            downsample (nn.Module, optional): Optional downsampling layer for skip connection
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            activation=nn.ReLU(inplace=True),
            use_bn=True
        )
        
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            activation=None,
            use_bn=True
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM layer.
    
    Attributes:
        lstm (nn.LSTM): LSTM layer
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True
    ):
        """
        Initialize the BiLSTM with specified parameters.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout probability
            batch_first (bool): Whether input/output has batch size as first dimension
        """
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=batch_first
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BiLSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, 2*hidden_size)
        """
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return x


class AttentionLayer(nn.Module):
    """
    Attention layer for focusing on relevant parts of the input.
    
    Attributes:
        query (nn.Linear): Query projection
        key (nn.Linear): Key projection
        value (nn.Linear): Value projection
        scale (float): Scaling factor for dot product
    """
    
    def __init__(self, input_dim: int, attention_dim: int):
        """
        Initialize the AttentionLayer with specified parameters.
        
        Args:
            input_dim (int): Dimension of input features
            attention_dim (int): Dimension of attention space
        """
        super(AttentionLayer, self).__init__()
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.scale = attention_dim ** 0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AttentionLayer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights
        """
        # Project inputs to queries, keys, and values
        q = self.query(x)  # (batch_size, seq_len, attention_dim)
        k = self.key(x)    # (batch_size, seq_len, attention_dim)
        v = self.value(x)  # (batch_size, seq_len, input_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)  # (batch_size, seq_len, input_dim)
        
        return output, attention_weights


class FeatureEncoder(nn.Module):
    """
    Encoder for audio features using CNN.
    
    Attributes:
        conv_layers (nn.ModuleList): List of convolutional blocks
        pool (nn.MaxPool2d): Max pooling layer
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_sizes: List[Union[int, Tuple[int, int]]],
        strides: List[Union[int, Tuple[int, int]]],
        paddings: List[Union[int, Tuple[int, int]]],
        dropout: float = 0.2
    ):
        """
        Initialize the FeatureEncoder with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions
            kernel_sizes (List[Union[int, Tuple[int, int]]]): List of kernel sizes
            strides (List[Union[int, Tuple[int, int]]]): List of stride values
            paddings (List[Union[int, Tuple[int, int]]]): List of padding values
            dropout (float): Dropout probability
        """
        super(FeatureEncoder, self).__init__()
        
        assert len(hidden_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All parameter lists must have the same length"
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_channels):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation=nn.ReLU(inplace=True),
                    use_bn=True
                )
            )
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeatureEncoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor
        """
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.conv_layers) - 1:  # No pooling after last conv
                x = self.pool(x)
            x = self.dropout(x)
        
        return x


class OnsetDetectionModule(nn.Module):
    """
    Module for detecting note onsets from audio features.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc (nn.Linear): Fully connected layer for onset prediction
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        dropout: float = 0.2
    ):
        """
        Initialize the OnsetDetectionModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            dropout (float): Dropout probability
        """
        super(OnsetDetectionModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layer for onset prediction
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OnsetDetectionModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            torch.Tensor: Onset probabilities of shape (batch_size, time_steps)
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply fully connected layer
        x = self.fc(x)  # (batch_size, time_steps', 1)
        x = x.squeeze(-1)  # (batch_size, time_steps')
        
        # Apply sigmoid to get probabilities
        x = torch.sigmoid(x)
        
        return x


class PitchDetectionModule(nn.Module):
    """
    Module for detecting pitch of notes from audio features.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc (nn.Linear): Fully connected layer for pitch prediction
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        num_pitches: int = 128,  # MIDI pitch range
        dropout: float = 0.2
    ):
        """
        Initialize the PitchDetectionModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            num_pitches (int): Number of possible pitches
            dropout (float): Dropout probability
        """
        super(PitchDetectionModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layer for pitch prediction
        self.fc = nn.Linear(lstm_hidden_size * 2, num_pitches)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PitchDetectionModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            torch.Tensor: Pitch probabilities of shape (batch_size, time_steps, num_pitches)
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply fully connected layer
        x = self.fc(x)  # (batch_size, time_steps', num_pitches)
        
        # Apply sigmoid to get probabilities
        x = torch.sigmoid(x)
        
        return x


class DurationEstimationModule(nn.Module):
    """
    Module for estimating duration of notes.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc (nn.Linear): Fully connected layer for duration prediction
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        max_duration_frames: int = 100,  # Maximum duration in frames
        dropout: float = 0.2
    ):
        """
        Initialize the DurationEstimationModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            max_duration_frames (int): Maximum duration in frames
            dropout (float): Dropout probability
        """
        super(DurationEstimationModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layer for duration prediction
        self.fc = nn.Linear(lstm_hidden_size * 2, max_duration_frames)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DurationEstimationModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            torch.Tensor: Duration probabilities of shape (batch_size, time_steps, max_duration_frames)
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply fully connected layer
        x = self.fc(x)  # (batch_size, time_steps', max_duration_frames)
        
        # Apply softmax to get probabilities
        x = F.softmax(x, dim=-1)
        
        return x


class TempoDetectionModule(nn.Module):
    """
    Module for detecting tempo from audio features.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc (nn.Linear): Fully connected layer for tempo prediction
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        dropout: float = 0.2
    ):
        """
        Initialize the TempoDetectionModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            dropout (float): Dropout probability
        """
        super(TempoDetectionModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layer for tempo prediction
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TempoDetectionModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            torch.Tensor: Tempo prediction of shape (batch_size, 1)
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, lstm_hidden_size*2)
        
        # Apply fully connected layer
        x = self.fc(x)  # (batch_size, 1)
        
        # Apply ReLU to ensure positive tempo
        x = F.relu(x)
        
        return x


class TimeSignatureModule(nn.Module):
    """
    Module for detecting time signature from audio features.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc_numerator (nn.Linear): Fully connected layer for numerator prediction
        fc_denominator (nn.Linear): Fully connected layer for denominator prediction
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        dropout: float = 0.2
    ):
        """
        Initialize the TimeSignatureModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            dropout (float): Dropout probability
        """
        super(TimeSignatureModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layers for time signature prediction
        self.fc_numerator = nn.Linear(lstm_hidden_size * 2, 8)  # Common numerators: 1-8
        self.fc_denominator = nn.Linear(lstm_hidden_size * 2, 4)  # Common denominators: 2, 4, 8, 16
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TimeSignatureModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Numerator and denominator probabilities
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, lstm_hidden_size*2)
        
        # Apply fully connected layers
        numerator = self.fc_numerator(x)  # (batch_size, 8)
        denominator = self.fc_denominator(x)  # (batch_size, 4)
        
        # Apply softmax to get probabilities
        numerator = F.softmax(numerator, dim=-1)
        denominator = F.softmax(denominator, dim=-1)
        
        return numerator, denominator


class InstrumentClassificationModule(nn.Module):
    """
    Module for classifying instruments in audio.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc (nn.Linear): Fully connected layer for instrument classification
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        num_instruments: int = 10,  # Number of instrument classes
        dropout: float = 0.2
    ):
        """
        Initialize the InstrumentClassificationModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            num_instruments (int): Number of instrument classes
            dropout (float): Dropout probability
        """
        super(InstrumentClassificationModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layer for instrument classification
        self.fc = nn.Linear(lstm_hidden_size * 2, num_instruments)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InstrumentClassificationModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            torch.Tensor: Instrument probabilities of shape (batch_size, num_instruments)
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, lstm_hidden_size*2)
        
        # Apply fully connected layer
        x = self.fc(x)  # (batch_size, num_instruments)
        
        # Apply sigmoid for multi-label classification
        x = torch.sigmoid(x)
        
        return x


class VelocityEstimationModule(nn.Module):
    """
    Module for estimating velocity of notes.
    
    Attributes:
        feature_encoder (FeatureEncoder): CNN encoder for audio features
        lstm (BiLSTM): Bidirectional LSTM for temporal modeling
        attention (AttentionLayer): Attention layer for focusing on relevant parts
        fc (nn.Linear): Fully connected layer for velocity prediction
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        lstm_hidden_size: int,
        attention_dim: int,
        dropout: float = 0.2
    ):
        """
        Initialize the VelocityEstimationModule with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            dropout (float): Dropout probability
        """
        super(VelocityEstimationModule, self).__init__()
        
        # CNN encoder for feature extraction
        self.feature_encoder = FeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 1, 1, 1],
            paddings=[1, 1, 1, 1],
            dropout=dropout
        )
        
        # Calculate output size of CNN
        self.cnn_output_channels = hidden_channels[-1]
        
        # BiLSTM for temporal modeling
        self.lstm = BiLSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            input_dim=lstm_hidden_size * 2,  # Bidirectional
            attention_dim=attention_dim
        )
        
        # Fully connected layer for velocity prediction
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VelocityEstimationModule.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            torch.Tensor: Velocity predictions of shape (batch_size, time_steps)
        """
        batch_size, _, time_steps, freq_bins = x.shape
        
        # Apply CNN encoder
        x = self.feature_encoder(x)  # (batch_size, cnn_output_channels, time_steps', freq_bins')
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch_size, time_steps', cnn_output_channels, freq_bins')
        x = x.reshape(batch_size, -1, self.cnn_output_channels)  # (batch_size, time_steps', cnn_output_channels)
        
        # Apply BiLSTM
        x = self.lstm(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply attention
        x, _ = self.attention(x)  # (batch_size, time_steps', lstm_hidden_size*2)
        
        # Apply fully connected layer
        x = self.fc(x)  # (batch_size, time_steps', 1)
        x = x.squeeze(-1)  # (batch_size, time_steps')
        
        # Apply sigmoid and scale to MIDI velocity range (0-127)
        x = torch.sigmoid(x) * 127
        
        return x


class MusicTranscriptionModel(nn.Module):
    """
    Complete model for music transcription combining all modules.
    
    Attributes:
        onset_detection (OnsetDetectionModule): Module for onset detection
        pitch_detection (PitchDetectionModule): Module for pitch detection
        duration_estimation (DurationEstimationModule): Module for duration estimation
        tempo_detection (TempoDetectionModule): Module for tempo detection
        time_signature (TimeSignatureModule): Module for time signature detection
        instrument_classification (InstrumentClassificationModule): Module for instrument classification
        velocity_estimation (VelocityEstimationModule): Module for velocity estimation
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: List[int] = [32, 64, 128, 256],
        lstm_hidden_size: int = 256,
        attention_dim: int = 128,
        num_pitches: int = 128,
        max_duration_frames: int = 100,
        num_instruments: int = 10,
        dropout: float = 0.2
    ):
        """
        Initialize the MusicTranscriptionModel with specified parameters.
        
        Args:
            input_channels (int): Number of input channels
            hidden_channels (List[int]): List of hidden channel dimensions for CNN
            lstm_hidden_size (int): Hidden size of LSTM
            attention_dim (int): Dimension of attention space
            num_pitches (int): Number of possible pitches
            max_duration_frames (int): Maximum duration in frames
            num_instruments (int): Number of instrument classes
            dropout (float): Dropout probability
        """
        super(MusicTranscriptionModel, self).__init__()
        
        # Onset detection module
        self.onset_detection = OnsetDetectionModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Pitch detection module
        self.pitch_detection = PitchDetectionModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            num_pitches=num_pitches,
            dropout=dropout
        )
        
        # Duration estimation module
        self.duration_estimation = DurationEstimationModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            max_duration_frames=max_duration_frames,
            dropout=dropout
        )
        
        # Tempo detection module
        self.tempo_detection = TempoDetectionModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Time signature module
        self.time_signature = TimeSignatureModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Instrument classification module
        self.instrument_classification = InstrumentClassificationModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            num_instruments=num_instruments,
            dropout=dropout
        )
        
        # Velocity estimation module
        self.velocity_estimation = VelocityEstimationModule(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            lstm_hidden_size=lstm_hidden_size,
            attention_dim=attention_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MusicTranscriptionModel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of predictions from all modules
        """
        # Apply all modules
        onset_probs = self.onset_detection(x)
        pitch_probs = self.pitch_detection(x)
        duration_probs = self.duration_estimation(x)
        tempo = self.tempo_detection(x)
        numerator_probs, denominator_probs = self.time_signature(x)
        instrument_probs = self.instrument_classification(x)
        velocity = self.velocity_estimation(x)
        
        # Return all predictions
        return {
            'onset_probs': onset_probs,
            'pitch_probs': pitch_probs,
            'duration_probs': duration_probs,
            'tempo': tempo,
            'numerator_probs': numerator_probs,
            'denominator_probs': denominator_probs,
            'instrument_probs': instrument_probs,
            'velocity': velocity
        }
    
    def predict(self, x: torch.Tensor, onset_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make predictions from audio features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, time_steps, freq_bins)
            onset_threshold (float): Threshold for onset detection
            
        Returns:
            Dict[str, Any]: Dictionary of processed predictions
        """
        # Get raw predictions
        predictions = self.forward(x)
        
        # Process onset predictions
        onset_probs = predictions['onset_probs']
        onsets = (onset_probs > onset_threshold).float()
        
        # Process pitch predictions
        pitch_probs = predictions['pitch_probs']
        
        # Process duration predictions
        duration_probs = predictions['duration_probs']
        durations = torch.argmax(duration_probs, dim=-1)
        
        # Process tempo prediction
        tempo = predictions['tempo']
        
        # Process time signature prediction
        numerator_probs = predictions['numerator_probs']
        denominator_probs = predictions['denominator_probs']
        numerator = torch.argmax(numerator_probs, dim=-1) + 1  # Add 1 because indices start from 0
        denominator_idx = torch.argmax(denominator_probs, dim=-1)
        denominator_values = torch.tensor([2, 4, 8, 16], device=denominator_idx.device)
        denominator = denominator_values[denominator_idx]
        
        # Process instrument prediction
        instrument_probs = predictions['instrument_probs']
        instruments = (instrument_probs > 0.5).float()
        
        # Process velocity prediction
        velocity = predictions['velocity']
        
        # Return processed predictions
        return {
            'onsets': onsets,
            'pitch_probs': pitch_probs,
            'durations': durations,
            'tempo': tempo,
            'numerator': numerator,
            'denominator': denominator,
            'instruments': instruments,
            'velocity': velocity,
            'raw_predictions': predictions
        }
