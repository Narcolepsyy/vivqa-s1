"""
Phrasal MCAN: MCAN with Phrasal Lexeme Module for Vietnamese VQA.

Integrates ViWordFormer's Phrasal Lexeme scoring into MCAN architecture:
- Phrasal Score computation via Bilinear layer
- Co-Text gated fusion for phrasal information propagation
- GPU-optimized implementation with matrix operations
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.base_classification import BaseClassificationModel
from utils.instance import Instance
from builders.encoder_builder import build_encoder
from builders.text_embedding_builder import build_text_embedding
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE

# Import to register phrasal encoders
import models.phrasal_mcan.attentions  # noqa: F401 - registers PhrasalScaledDotProductAttention
import models.phrasal_mcan.encoder      # noqa: F401 - registers PhrasalEncoder, PhrasalGuidedAttentionEncoder


class MLP(nn.Module):
    """Simple MLP for attention reduction."""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(config.D_MODEL, 1)

    def forward(self, features: torch.Tensor):
        output = self.dropout(self.relu(self.fc1(features)))
        output = self.fc2(output)
        return output


@META_ARCHITECTURE.register()
class PhrasalMCAN(BaseClassificationModel):
    """
    Phrasal MCAN: MCAN with Phrasal Lexeme scoring for Vietnamese VQA.
    
    Architecture:
    1. Text Embedding: Syllable-level embedding (Vietnamese text)
    2. Vision Embedding: Region features from pre-trained detector
    3. Self-Encoder: PhrasalEncoder with phrasal-aware self-attention
    4. Guided-Encoder: PhrasalGuidedAttentionEncoder for cross-modal fusion
    5. Classification: MLP with softmax over answer vocabulary
    
    Key Features:
    - Phrasal Score: Bilinear(x_i, x_j) captures Vietnamese phrase structure
    - Co-Text Gating: Preserves phrasal information across deep layers
    - GPU-optimized: All operations use matrix ops, no for-loops
    """

    def __init__(self, config, vocab):
        super().__init__(config, vocab)

        self.device = torch.device(config.DEVICE)

        # Embeddings
        self.text_embedding = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)

        # Phrasal-aware Encoders
        self.self_encoder = build_encoder(config.SELF_ENCODER)
        self.guided_encoder = build_encoder(config.GUIDED_ENCODER)

        # Attention reduction for pooling
        self.vision_attr_reduce = MLP(config.VISION_ATTR_REDUCE)
        self.text_attr_reduce = MLP(config.TEXT_ATTR_REDUCE)

        # Feature projection and fusion
        self.vision_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.text_proj = nn.Linear(config.D_MODEL, config.D_MODEL)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        # Classification head
        self.classify = nn.Linear(config.D_MODEL, vocab.total_answers)

    def forward(self, input_features: Instance):
        """
        Forward pass for Vietnamese VQA.
        
        Args:
            input_features: Instance containing:
                - region_features: Visual features (batch, n_regions, d_feature)
                - question_tokens: Tokenized Vietnamese question (syllable-level)
        
        Returns:
            Log-softmax scores over answer vocabulary
        """
        # Vision embedding
        vision_features = input_features.region_features
        vision_features, vision_padding_mask = self.vision_embedding(vision_features)

        # Text embedding (syllable-level for Vietnamese)
        question_tokens = input_features.question_tokens
        text_features, (text_padding_mask, _) = self.text_embedding(question_tokens)
        
        # Self-Attention encoder with Phrasal scoring (SA)
        text_features = self.self_encoder(
            features=text_features,
            padding_mask=text_padding_mask
        )

        # Guided Self-Attention encoder (GSA) - vision guided by language
        vision_features = self.guided_encoder(
            vision_features=vision_features,
            vision_padding_mask=vision_padding_mask,
            language_features=text_features,
            language_padding_mask=text_padding_mask
        )

        # Attention pooling
        attended_vision_features = self.vision_attr_reduce(vision_features)
        attended_vision_features = F.softmax(attended_vision_features, dim=1)
        attended_text_features = self.text_attr_reduce(text_features)
        attended_text_features = F.softmax(attended_text_features, dim=1)

        # Weighted feature aggregation
        weighted_vision_features = (vision_features * attended_vision_features).sum(dim=1)
        weighted_text_features = (text_features * attended_text_features).sum(dim=1)

        # Feature fusion and classification
        output = self.layer_norm(
            self.vision_proj(weighted_vision_features) + 
            self.text_proj(weighted_text_features)
        )
        output = self.classify(output)

        return F.log_softmax(output, dim=-1)
