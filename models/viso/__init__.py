# Importing this package registers ViSO and FlatS1 in META_ARCHITECTURE (train_viso.py imports it explicitly).
from .encoder import ViSOEncoder
from .heads import ChoiceHead, EscalationHead, NoulHead, ScoreHead, SetAttention
from .viso_model import FlatS1, ViSO
from .butd import BUTD, ButdEncoder
from .viso_ft import ViSOFT, LiveEncoders
