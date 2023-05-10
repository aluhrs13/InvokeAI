from typing import Literal
from pydantic import Field
from .baseinvocation import BaseInvocation, InvocationContext
from .prompt import PromptOutput


from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

class DeepFloydInvocation(BaseInvocation):
    """Just passthrough the prompt"""
    #fmt: off
    type: Literal["deep_floyd"] = "deep_floyd"
    prompt: str = Field(default=None, description="The input prompt")
    negative_prompt: str = Field(default=None, description="The input prompt")
    stage_1_model: str = Field(default="DeepFloyd/IF-I-XL-v1.0", description="The stage1 model")
    stage_2_model: str = Field(default="DeepFloyd/IF-II-L-v1.0", description="The stage2 model")
    stage_3_model: str = Field(default="stabilityai/stable-diffusion-x4-upscaler", description="The stage3 model")
    variant: str = Field(default="fp16", description="The model variant")
    noise_level: int = Field(default=100, description="The noise level")
    enable_cpu_offload: bool = Field(default=True, description="Enable CPU offload")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        #TODO: Seed?
        #TODO: text_encoder?
        #TODO: Safety modules?
        
        #TODO: Do this right.
        dtype = torch.float16 if self.variant == "fp16" else torch.float32

        # stage 1
        stage_1 = DiffusionPipeline.from_pretrained(self.stage_1_model, variant=self.variant, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_1.enable_model_cpu_offload()

        # stage 2
        stage_2 = DiffusionPipeline.from_pretrained(self.stage_2_model, text_encoder=None, variant=self.variant, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_2.enable_model_cpu_offload()

        # stage 3
        safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
        stage_3 = DiffusionPipeline.from_pretrained(self.stage_3_model, **safety_modules, torch_dtype=dtype)

        if self.enable_cpu_offload:
            stage_3.enable_model_cpu_offload()

        # text embeds
        #TODO: Negative Prompt
        prompt_embeds, negative_embeds = stage_1.encode_prompt(self.prompt)

        generator = torch.manual_seed(0)

        image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        pt_to_pil(image)[0].save("./if_stage_I.png")


        image = stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images
        pt_to_pil(image)[0].save("./if_stage_II.png")


        image = stage_3(prompt=self.prompt, image=image, generator=generator, noise_level=100).images
        image[0].save("./if_stage_III.png")


        return PromptOutput(prompt=self.prompt)





