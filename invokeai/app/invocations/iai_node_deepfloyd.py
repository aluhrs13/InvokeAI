from typing import Literal, Optional
from pydantic import BaseModel, Field
from .baseinvocation import BaseInvocation, InvocationContext, BaseInvocationOutput
from .image import ImageOutput, build_image_output

from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

from ..models.image import ImageField, ImageType
from .latent import LatentsOutput, LatentsField


"""
TODO:
- Can each node take in noise, or does that need be a generator?
- Pass the prompt_embeds between stage 1 and stage 2
- Is stage 3 just resize latents?
- Get seed from latent.py

Bigger TODO: Can this all be done in latent.py?
"""

"""
class GeneratePromptEmbeds(BaseInvocation):

    def invoke(self, context: InvocationContext) -> LatentOutput:
        return LatentOutput(latent=torch.randn(1, 512, 1, 1))
"""
class DeepFloydStage1Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_1"] = "deep_floyd_stage_1"
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The input prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage1 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        dtype = torch.float16
        variant = "fp16"

        # stage 1
        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype)
        
        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()

        prompt_embeds, negative_embeds = stage_1.encode_prompt(self.prompt)

        generator = torch.manual_seed(0)
        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        pt_to_pil(image)[0].save("./if_stage_I.png")

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, image)
        return LatentsOutput(latents=LatentsField(latents_name=name))


class DeepFloydStage2Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_2"] = "deep_floyd_stage_2"
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The input prompt")
    latents: Optional[LatentsField] = Field(description="The latents to generate an image from")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage1 model")
    stage_2_model: str = Field(default="DeepFloyd/IF-II-L-v1.0", description="The stage2 model")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        dtype = torch.float16
        variant = "fp16"
        latents = context.services.latents.get(self.latents.latents_name)

        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=variant, torch_dtype=dtype)
        
        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()

        prompt_embeds, negative_embeds = stage_1.encode_prompt(self.prompt)

        # stage 2
        stage_2 = DiffusionPipeline.from_pretrained(self.stage_2_model, text_encoder=None, variant=variant, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_2.enable_model_cpu_offload()

        generator = torch.manual_seed(0)


        image = stage_2(
            image=latents, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images
        pt_to_pil(image)[0].save("./if_stage_II.png")


        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, image)
        return LatentsOutput(latents=LatentsField(latents_name=name))

class DeepFloydStage3Invocation(BaseInvocation):
    #fmt: off
    type: Literal["deep_floyd_stage_3"] = "deep_floyd_stage_3"
    latents: Optional[LatentsField] = Field(description="The latents to generate an image from")
    prompt: str = Field(default=None, description="The input prompt")
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

        generator = torch.manual_seed(0)

        #TODO: Negative prompt?
        image = stage_3(prompt=self.prompt, image=latents, generator=generator, noise_level=self.noise_level).images
        image[0].save("./if_stage_III.png")

        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )

        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )

        torch.cuda.empty_cache()

        context.services.images.save(image_type, image_name, image[0], metadata)
        return build_image_output(
            image_type=image_type, image_name=image_name, image=image[0]
        )



