# Source Map Generator 

An exploration into machine learning and Source Map Format.

## Prerequisites

```
pip install transformers datasets accelerate
```

### CUDA
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is enabled:
```
import torch
print(torch.cuda.is_available())  # Should return True
```

### BSPsrc
```
git clone https://github.com/ata4/bspsrc
```

## How it works

1. Modify and run the ./bsp/batch_convert_to_vmf.py script to decompile Garry's Mod maps into VMF files
2. Run the ./vmf/tokenize_vmf.py script to tokenize the VMF files into training data.
3. Train the model by running ./training/train_model.py