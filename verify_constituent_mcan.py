import torch
import numpy as np
from types import SimpleNamespace
from models.constituent_mcan.mcan_constituent import ConstituentMCAN
from utils.instance import Instance

def test_constituent_mcan():
    # Mock Config
    config = SimpleNamespace(
        DEVICE="cpu",
        DROPOUT=0.1,
        D_MODEL=512,
        CONSTITUENT_LAYERS=2,
        TEXT_EMBEDDING=SimpleNamespace(
            ARCHITECTURE="UsualEmbedding",
            D_MODEL=512,
            WORD_EMBEDDING=None,
            DROPOUT=0.1
        ),
        VISION_EMBEDDING=SimpleNamespace(
            ARCHITECTURE="FeatureEmbedding",
            D_MODEL=512,
            D_FEATURE=2048,
            DROPOUT=0.1
        ),
        SELF_ENCODER=SimpleNamespace(
            ARCHITECTURE="Encoder",
            LAYERS=2,
            D_MODEL=512,
            SELF_ATTENTION=SimpleNamespace(
                ARCHITECTURE="ScaledDotProductAttention",
                HEAD=8,
                D_MODEL=512,
                D_KEY=64,
                D_VALUE=64,
                D_FF=2048,
                DROPOUT=0.1,
                USE_AOA=False,
                CAN_BE_STATEFUL=False
            )
        ),
        GUIDED_ENCODER=SimpleNamespace(
            ARCHITECTURE="GuidedAttentionEncoder",
            LAYERS=2,
            D_MODEL=512,
            GUIDED_ATTENTION=SimpleNamespace(
                ARCHITECTURE="ScaledDotProductAttention",
                HEAD=8,
                D_MODEL=512,
                D_KEY=64,
                D_VALUE=64,
                D_FF=2048,
                DROPOUT=0.1,
                USE_AOA=False,
                CAN_BE_STATEFUL=False
            )
        ),
        VISION_ATTR_REDUCE=SimpleNamespace( D_MODEL=512, DROPOUT=0.1 ),
        TEXT_ATTR_REDUCE=SimpleNamespace( D_MODEL=512, DROPOUT=0.1 )
    )

    # Mock Vocab
    class MockVocab:
        def __init__(self):
            self.total_answers = 1000
            self.stoi = {"<pad>": 0}
            self.padding_token = "<pad>"
            self.padding_idx = 0
        def __len__(self):
            return 100
    
    vocab = MockVocab()

    # Instantiate Model
    print("Instantiating ConstituentMCAN...")
    try:
        model = ConstituentMCAN(config, vocab)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error during instantiation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Mock Input
    batch_size = 2
    seq_len = 15
    num_regions = 36
    
    input_features = Instance(
        region_features=torch.randn(batch_size, num_regions, 2048),
        question_tokens=torch.randint(0, 100, (batch_size, seq_len))
    )

    # Forward Pass
    print("Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(input_features)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
        expected_shape = (batch_size, vocab.total_answers)
        if output.shape == expected_shape:
            print("Output shape is correct.")
        else:
            print(f"Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_constituent_mcan()
