from typing import Literal, Optional
from pydantic import BaseModel, Field
from .baseinvocation import BaseInvocation, InvocationContext, BaseInvocationOutput
from .image import ImageOutput, build_image_output

from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

from ..models.image import ImageType
from .latent import LatentsOutput, LatentsField, random_seed

"""
TODO:
- Can each node take in noise, or does that need be a generator?
- Figure out if stage 3 can be implemented generically and contributed to main
- Get seed from latent.py or random number node from main
- i2i and inpainting
- How to do dtype and variant correctly
- Use text encoder raw for PromptEmbeds? https://huggingface.co/blog/if has an example, but needs bitsandbytes?
- Add comment/readme with directions for license and pip requirements
- Select box for model
- Progress images?
- Schema_extra stuff
- Size input?
"""

class LatentsPairOutput(BaseInvocationOutput):
    # fmt: off
    type: Literal["latents_pair"] = "latents_pair"
    latents1: LatentsField = Field(default=None, description="Latents #1")
    latents2: LatentsField = Field(default=None, description="Latents #2")
    # fmt: on

class PromptEmbedsInvocation(BaseInvocation):
    type: Literal["prompt_embeds"] = "prompt_embeds"
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The negative prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage 1 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")

    def invoke(self, context: InvocationContext) -> LatentsPairOutput:
        dtype = torch.float16
        variant = "fp16"

        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype)
        
        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()

        #TODO: Actually include negative prompt
        prompt_embeds, negative_embeds = stage_1.encode_prompt(self.prompt)

        name1 = f'{context.graph_execution_state_id}__{self.id}_prompt'
        context.services.latents.set(name1, prompt_embeds)

        name2 = f'{context.graph_execution_state_id}__{self.id}_negative_prompt'
        context.services.latents.set(name2, negative_embeds)

        return LatentsPairOutput(latents1=LatentsField(latents_name=name1), latents2=LatentsField(latents_name=name2))

class DeepFloydStage1Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_1"] = "deep_floyd_stage_1"
    prompt_embeds: LatentsField = Field(default=None, description="The input prompt")
    negative_embeds: LatentsField = Field(default=None, description="The input negative prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage1 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        dtype = torch.float16
        variant = "fp16"
        prompt_embeds = context.services.latents.get(self.prompt_embeds.latents_name)
        negative_embeds = context.services.latents.get(self.negative_embeds.latents_name)

        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype)
        
        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()

        #TODO: Seed
        generator = torch.manual_seed(0)
        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, image)
        return LatentsOutput(latents=LatentsField(latents_name=name))


class DeepFloydStage2Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_2"] = "deep_floyd_stage_2"
    prompt_embeds: LatentsField = Field(default=None, description="The input prompt")
    negative_embeds: LatentsField = Field(default=None, description="The input negative prompt")
    latents: Optional[LatentsField] = Field(description="The latents to generate an image from")
    stage_2_model: str = Field(default="DeepFloyd/IF-II-L-v1.0", description="The stage2 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        dtype = torch.float16
        variant = "fp16"
        latents = context.services.latents.get(self.latents.latents_name)
        prompt_embeds = context.services.latents.get(self.prompt_embeds.latents_name)
        negative_embeds = context.services.latents.get(self.negative_embeds.latents_name)

        stage_2 = DiffusionPipeline.from_pretrained(self.stage_2_model, text_encoder=None, variant=variant, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_2.enable_model_cpu_offload()

        #TODO: Seed
        generator = torch.manual_seed(0)

        image = stage_2(
            image=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, image)
        return LatentsOutput(latents=LatentsField(latents_name=name))

class DeepFloydStage3Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_3"] = "deep_floyd_stage_3"
    latents: Optional[LatentsField] = Field(description="The latents to generate an image from")
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The negative prompt")
    stage_3_model: str = Field(default="stabilityai/stable-diffusion-x4-upscaler", description="The stage3 model")
    noise_level: int = Field(default=100, description="The noise level")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        dtype = torch.float16
        latents = context.services.latents.get(self.latents.latents_name)

        # stage 3
        # safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained(self.stage_3_model, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_3.enable_model_cpu_offload()

        #TODO: Seed
        generator = torch.manual_seed(0)

        #TODO: Negative prompt
        image = stage_3(prompt=self.prompt, image=latents, generator=generator, noise_level=self.noise_level).images


        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, image[0], metadata)

        #TODO: Should I be doing this elsewhere too?
        torch.cuda.empty_cache()

        return build_image_output(
            image_type=image_type, image_name=image_name, image=image[0]
        )



