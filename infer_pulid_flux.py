import time
import numpy as np  
import torch
from einops import rearrange
from PIL import Image
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip

class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args.fp8,
        )
        self.pulid_model = PuLIDPipeline(self.model, device="cpu" if offload else device, weight_dtype=torch.bfloat16,
                                         onnx_provider=args.onnx_provider)
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args.pretrained_model)

    @torch.inference_mode()
    def generate_image(self, width, height, num_steps, start_step, guidance, seed, prompt, id_image=None, id_weight=1.0, neg_prompt="", true_cfg=1.0, timestep_to_start_cfg=1, max_sequence_length=128):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image_np = np.array(id_image)

            id_image_resized = resize_numpy_image_long(id_image_np, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image_resized, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="PuLID for FLUX.1-dev")
    parser.add_argument("--name", type=str, default="flux-dev", help="currently only support flux-dev")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use, for 24G GPUs")
    parser.add_argument("--fp8", action="store_true", help="use flux-dev-fp8 model")
    parser.add_argument("--onnx_provider", type=str, default="gpu", help="ONNX provider (gpu/cpu)")
    parser.add_argument("--pretrained_model", type=str, help="for development")
    parser.add_argument("--width", type=int, default=896, help="Width of the output image")
    parser.add_argument("--height", type=int, default=1152, help="Height of the output image")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--start_step", type=int, default=0, help="Start step for inserting ID")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--prompt", type=str, default="portrait, color, cinematic", help="Text prompt for the image generation")
    parser.add_argument("--id_image_path", type=str, default="example_inputs/liuyifei.png", help="Path to the ID image")
    parser.add_argument("--id_weight", type=float, default=1.0, help="Weight of the ID image")
    parser.add_argument("--neg_prompt", type=str, default="bad quality, worst quality", help="Negative prompt for the image generation")
    parser.add_argument("--true_cfg", type=float, default=1.0, help="True CFG scale")
    parser.add_argument("--timestep_to_start_cfg", type=int, default=1, help="Timestep to start CFG")
    parser.add_argument("--max_sequence_length", type=int, default=128, help="Max sequence length for T5 prompt")

    args = parser.parse_args()

    generator = FluxGenerator(args.name, args.device, args.offload, args.aggressive_offload, args)

    # Load the ID image
    id_image = Image.open(args.id_image_path)

    # Generate the image
    img, seed = generator.generate_image(
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        start_step=args.start_step,
        guidance=args.guidance,
        seed=args.seed,
        prompt=args.prompt,
        id_image=id_image,
        id_weight=args.id_weight,
        neg_prompt=args.neg_prompt,
        true_cfg=args.true_cfg,
        timestep_to_start_cfg=args.timestep_to_start_cfg,
        max_sequence_length=args.max_sequence_length
    )

    # Save the generated image
    img.save(f"generated_image_{seed}.png")
    print(f"Image saved as 'generated_image_{seed}.png'")
