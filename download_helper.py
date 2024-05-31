# Download the models to the ./torch

import os
import torch
from lavis.models import load_model_and_preprocess

os.environ['TORCH_HOME']="./torch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model1, vis_processors1, _                = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device) 
 
model2, vis_processors2, text_processors2 = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
 
model3, vis_processors3, text_processors3 = load_model_and_preprocess(
            name="blip_image_text_matching", model_type="base", is_eval=True, device=device)