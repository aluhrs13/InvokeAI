from typing import Literal, Optional
from pydantic import BaseModel, Field
from .baseinvocation import BaseInvocation, InvocationContext, BaseInvocationOutput
from .image import ImageOutput

from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

from ..models.image import ImageField, ImageType

class FloatTensorField(BaseModel):
    """A latents field used for passing latents between invocations"""

    float_tensor_name: Optional[str] = Field(default=None, description="The name of the float tensor")

    class Config:
        schema_extra = {"required": ["float_tensor_name"]}

class DeepFloydPipelineOutput(BaseInvocationOutput):
    """A collection of integers"""
    type: Literal["deep_floyd_transfer"] = "deep_floyd_transfer"

    # Outputs
    image: ImageField = Field(default=[], description="The image")
    prompt_embeds: FloatTensorField = Field(default=None, description="embeds")
    negative_embeds: FloatTensorField = Field(default=None, description="neg embeds")
    seed: int = Field(default=None, description="generator seed")

    class Config:
        schema_extra = {
            "required": ["image", "prompt_embeds", "negative_embeds", "generator"]
        }

class DeepFloydStage1Invocation(BaseInvocation):
    """Stage 1 of a DeepFloyd Pipeline"""
    #fmt: off
    type: Literal["deep_floyd_stage_1"] = "deep_floyd_stage_1"
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The input prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage1 model")
    noise_level: int = Field(default=100, description="The noise level")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> DeepFloydPipelineOutput:
        #TODO: Seed?
        #TODO: text_encoder?
        #TODO: Safety modules?
        
        dtype = torch.float16
        variant = "fp16"

        # stage 1
        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()

        #TODO: Negative Prompt
        prompt_embeds, negative_embeds = stage_1.encode_prompt(self.prompt, self.negative_prompt)
        generator = torch.manual_seed(0)
        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        pt_to_pil(image)[0].save("./if_stage_I.png")


        return DeepFloydPipelineOutput(image=image, prompt_embeds=prompt_embeds, negative_embeds=negative_embeds)


class DeepFloydStage2Invocation(BaseInvocation):
    """Just passthrough the prompt"""
    #fmt: off
    type: Literal["deep_floyd_stage_2"] = "deep_floyd_stage_2"
    dftransfer: DeepFloydPipelineOutput = Field(default=None, description="dftransfer")
    stage_2_model: str = Field(default="DeepFloyd/IF-II-L-v1.0", description="The stage2 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        #TODO: Seed?
        #TODO: text_encoder?
        
        dtype = torch.float16
        variant = "fp16"

        # stage 2
        stage_2 = DiffusionPipeline.from_pretrained(self.stage_2_model, text_encoder=None, variant=variant, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_2.enable_model_cpu_offload()

        generator = torch.manual_seed(0)


        image = stage_2(
            image=image, prompt_embeds=self.dftransfer.prompt_embeds, negative_prompt_embeds=self.dftransfer.negative_embeds, generator=generator, output_type="pt"
        ).images
        pt_to_pil(image)[0].save("./if_stage_II.png")


        return ImageOutput(prompt=self.prompt)


class DeepFloydStage3Invocation(BaseInvocation):
    """Just passthrough the prompt"""
    #fmt: off
    type: Literal["deep_floyd_stage_3"] = "deep_floyd_stage_3"
    prompt: str = Field(default=None, description="The input prompt")
    stage_3_model: str = Field(default="stabilityai/stable-diffusion-x4-upscaler", description="The stage3 model")
    noise_level: int = Field(default=100, description="The noise level")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        #TODO: Seed?
        #TODO: Safety modules?
        
        dtype = torch.float16

        # stage 3
        # safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained(self.stage_3_model, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_3.enable_model_cpu_offload()

        generator = torch.manual_seed(0)

        #TODO: Negative prompt?
        image = stage_3(prompt=self.prompt, image=image, generator=generator, noise_level=self.noise_level).images
        image[0].save("./if_stage_III.png")

        return ImageOutput(prompt=self.prompt)




