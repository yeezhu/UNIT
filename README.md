# UNIT: Unifying Image and Text Recognition in One Vision Encoder


## Install

```Shell
pip install torch==2.1.0 
pip install timm==0.9.12 
pip install transformers==4.32.1
    
```

This project supports both NVIDIA and Ascend GPUs.



## Usage

```Python
from PIL import Image
from transformers import CLIPImageProcessor

from unit import UNITModel

model_path = "/path/to/UNIT_600M/"

model = UNITModel.from_pretrained(model_path)

model.to(device='cuda')
model.eval()

image_processor = CLIPImageProcessor.from_pretrained(model_path)

image = Image.open("test.jpg").convert('RGB')

image_input = image_processor(image)['pixel_values'][0]
image_tensor = torch.tensor(image_input).unsqueeze(0).to(torch.bfloat16).cuda()

with torch.set_grad_enabled(False):
    cls_tokens, spatial_tokens = model(image_tensor)

```

## Model Zoo

Please refer to here to obtain our pretrained UNIT_600M model.