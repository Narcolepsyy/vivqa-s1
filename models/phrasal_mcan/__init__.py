# Phrasal MCAN module
# Integrates Phrasal Lexeme scoring from ViWordFormer into MCAN architecture

from .attentions import PhrasalScaledDotProductAttention
from .layers import PhrasalMultiHeadAttention, PhrasalEncoderLayer
from .encoder import PhrasalEncoder, PhrasalGuidedAttentionEncoder
from .mcan_phrasal import PhrasalMCAN
