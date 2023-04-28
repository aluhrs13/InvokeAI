# Step 1 - Mostly copy/pasting things into the right places:
#    - Imports at the top
#    - Args into inputs
#    - Functions into the class
#    - __main__ into invoke()
#    - add self. to variables that need it.
# python -m pip install -e segment_anything
# python -m pip install -e GroundingDINO
# git submodule update --init --recursive
# cd Tag2Text && pip install -r requirements.txt
from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from .image import ImageField, ImageOutput, ImageType

import argparse
import os
import copy

import numpy as np
import json
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import nltk

import sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
print(sys.path)
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

class GroundedSegmentAnythingInvocation(BaseInvocation):
    """Use grounded segment anything to make a mask - https://github.com/IDEA-Research/Grounded-Segment-Anything"""
    #fmt: off
    type: Literal["grounded_segment_anything"] = "grounded_segment_anything"
    config_file: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\GroundingDINO_SwinT_OGC.py", description="path to config file")  # change the path of the model config file
    tag2text_checkpoint: str = Field(default="E:\\StableDiffusion\\Tag2Text\\tag2text_swin_14m.pth", description="path to checkpoint file")
    grounded_checkpoint: str = Field(default="E:\\StableDiffusion\\GroundingDINO\\groundingdino_swint_ogc.pth", description="path to checkpoint filet")
    sam_checkpoint: str = Field(default="E:\\StableDiffusion\\SegmentAnything\\sam_vit_h_4b8939.pth", description="path to checkpoint file")
    image: ImageField = Field(default=None, description="The image to run inference on.")
    split: str = Field(default=",", description="split for text prompt")
    output_dir: str = Field(default="E:\\StableDiffusion", description="output directory")
    box_threshold: float = Field(default=0.25, description="box threshold")
    text_threshold: float = Field(default=0.2, description="text treshold")
    iou_threshold: float = Field(default=0.5, description="iou threshold")
    cpu_only: bool = Field(default=True, description="running on cpu only!, default=False")
    #fmt: on


    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image


    def generate_caption(self, raw_image, processor, blip_model, device):
        # unconditional image captioning
        if device == "cuda":
            inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
        else:
            inputs = processor(raw_image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption


    def generate_tags(self, caption, split=',', max_tokens=100, model="gpt-3.5-turbo"):
        lemma = nltk.wordnet.WordNetLemmatizer()
        nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])
        tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == 'N']
        tags_lemma = [lemma.lemmatize(w) for w in tags_list]
        tags = ', '.join(map(str, tags_lemma))
        return tags


    #TODO: Might not need this without chatGPT
    def check_caption(self, caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
        object_list = [obj.split('(')[0] for obj in pred_phrases]
        object_num = []
        for obj in set(object_list):
            object_num.append(f'{object_list.count(obj)} {obj}')
        object_num = ', '.join(object_num)
        print(f"Correct object number: {object_num}")
        return caption



    def load_model(self, model_config_path, model_checkpoint_path, cpu_only=False):
        args = SLConfig.fromfile(model_config_path)
        args.device = "cuda" if not cpu_only else "cpu"
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold,device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases


    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)


    def save_mask_data(self, output_dir, caption, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = {
            'caption': caption,
            'mask':[{
                'value': value,
                'label': 'background'
            }]
        }
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data['mask'].append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'label.json'), 'w') as f:
            json.dump(json_data, f)

    def invoke(self, context: InvocationContext) -> ImageOutput:        
        # make dir
        os.makedirs(self.output_dir, exist_ok=True)
        # load image
        initial_image = context.services.images.get(
            self.image.image_type, self.image.image_name
        )

        image_pil = initial_image.copy()
        
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)

        # load model
        model = self.load_model(self.config_file, self.grounded_checkpoint, cpu_only=self.cpu_only)

        # visualize raw image
        image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))

        # generate caption and tags
        # use Tag2Text can generate better captions
        # https://huggingface.co/spaces/xinyu1205/Tag2Text
        # but there are some bugs...
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        if self.cpu_only == False:
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
            device = "cuda"
        else:
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            device="cpu"
        caption = self.generate_caption(image_pil, processor, blip_model, device=device)
        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        text_prompt = self.generate_tags(caption, split=self.split)
        print(f"Caption: {caption}")
        print(f"Tags: {text_prompt}")

        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            model, image, text_prompt, self.box_threshold, self.text_threshold, device=device
        )

        # initialize SAM
        predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(device))
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")
        caption = self.check_caption(caption, pred_phrases)
        print(f"Revise caption with number: {caption}")

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )
        
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.title(caption)
        plt.axis('off')
        plt.savefig(
            os.path.join(self.output_dir, "automatic_label_output.jpg"), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        self.save_mask_data(self.output_dir, caption, masks, boxes_filt, pred_phrases)

        image_type = ImageType.INTERMEDIATE
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        context.services.images.save(image_type, image_name, image_pil, metadata)
        return ImageOutput(image=ImageField(image_type=image_type, image_name=image_name))