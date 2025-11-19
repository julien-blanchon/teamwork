from pathlib import Path
import os
import torch
from tqdm import tqdm
from teamwork import TeamworkPipeline
from gfxds import Loader, save_image, open_image
import gc
import json
import typer
import numpy as np
import traceback
from einops import rearrange
from typing import Annotated
import math
import re

app = typer.Typer()


@app.command()
@torch.no_grad()
def generate(
    output: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path, typer.Option()],
    dataset: Annotated[str, typer.Option()],
    replicates: int = 1,
    limit: int | None = None,
    input_teammate: str = "image",
    captions: bool = False,
    compile: bool = False,
    exist_ok: bool = False,
):
    args = locals()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pipe = TeamworkPipeline.from_checkpoint(checkpoint, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    if not captions:
        pipe.empty_prompts()
    gc.collect()
    torch.cuda.empty_cache()
    print(pipe.teamwork_config)

    # Get the dataset
    if dataset.strip().startswith("{"):
        dataset_cfg = json.loads(dataset)
    else:
        dataset_cfg = dataset
    loader = Loader("./datasets", dataset_cfg, concurrent=8, limit=limit)
    loader.start()
    print(f"Found {len(loader)} eval samples")

    output.mkdir(exist_ok=exist_ok)
    (output / "arguments.txt").write_text(
        "\n".join(f"--{a} {v}" for a, v in args.items())
    )

    # Main inference loop
    for index, batch in enumerate(tqdm(loader, total=len(loader))):
        name = batch.name or str(index)
        for c, r in batch.images.items():
            save_image(output / f"{name}.gt" / f"{c}.png", r.image)
        if input_teammate in batch.images:
            input_image = batch.images[input_teammate].image
        else:
            input_image = batch.images["render"].image
        for replicate in range(replicates):
            outputs = pipe(
                {input_teammate: input_image},
                prompt=batch.caption if captions else None,
                output_type="np",
            )
            for component, image in derive_more(outputs, input_image).items():
                save_image(
                    output / f"{name}.{replicate:03}" / f"{component}.png", image
                )


def derive_more(d: dict[str, np.ndarray], image: np.ndarray) -> dict[str, np.ndarray]:
    more = d.copy()
    derive_diffuse = "albedo" in d and "metalness" in d
    derive_albedo = "specular" in d and "diffuse" in d

    if derive_diffuse:
        metalness = d["metalness"]
        specular_base = (1 - metalness) * 0.04
        albedo_base = d["albedo"] ** 2.2 - specular_base
        diffuse = (albedo_base.clip(0, 1) * (1 - metalness)) ** (1 / 2.2)
        specular = (albedo_base * metalness + specular_base) ** (1 / 2.2)
        if "diffuse" not in d:
            d["diffuse"] = diffuse
        more["diffuse.der"] = diffuse
        if "specular" not in d:
            d["specular"] = specular
        more["specular.der"] = specular

    if derive_albedo:
        albedo = (d["diffuse"] ** 2.2 + d["specular"] ** 2.2) ** (1 / 2.2)
        if "albedo" not in d:
            d["albedo"] = albedo
        more["albedo.der"] = albedo

    if "albedo" in d:
        albedo = d["albedo"] ** 2.2
        image = image**2.2
        inverseshading = image / (albedo + image)
        inverseshading = inverseshading ** (1 / 2.2)
        more["inverseshading.der"] = inverseshading

    if "albedo" in d and "inverseshading" in d:
        albedo = d["albedo"] ** 2.2
        shading = d["inverseshading"] ** 2.2
        shading = shading.clip(0.0, 0.9999)
        shading = shading / (1 - shading)
        more["image.der2"] = (albedo * shading) ** (1 / 2.2)

    if "diffuse" in d and "diffuseshading" in d and "residual" in d:
        diffuse = d["diffuse"] ** 2.2
        shading = d["diffuseshading"] ** 2.2
        residual = d["residual"] ** 2.2
        more["image.der3"] = (diffuse * shading + residual) ** (1 / 2.2)

    return more


def lsq_whitepoint(test, result):
    factor = (test * test).sum(axis=(0, 1), keepdims=True) / (test * result).sum(
        axis=(0, 1), keepdims=True
    )
    return result * factor


def rms_error(test_chans, result_chans):
    se = (test_chans - result_chans) ** 2
    mse = se.mean().item()
    return math.sqrt(mse)


@app.command()
@torch.no_grad()
def score(
    output: Annotated[list[Path], typer.Option()],
    gt_prefix: Path | None = None,
    input_component: str = "render",
):
    subdirs: list[Path] = []
    dir: Path
    for dir in output:
        subdirs += [
            d for d in dir.iterdir() if d.is_dir() and not d.name.endswith(".gt")
        ]

    np.seterr(divide="raise")

    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_torch = LearnedPerceptualImagePatchSimilarity(net_type="alex").to("cuda")

        def lpips_error(a, b):
            a = (
                rearrange(torch.tensor(a, device="cuda"), "w h c -> c w h").unsqueeze(0)
                * 2
                - 1
            )
            b = (
                rearrange(torch.tensor(b, device="cuda"), "w h c -> c w h").unsqueeze(0)
                * 2
                - 1
            )
            return lpips_torch(a, b).item()
    except ImportError:
        print("Skipping LPIPS error: torchmetrics not installed")
        lpips_error = None

    for subdir in tqdm(subdirs):
        gtdir_name, subcount = re.subn(r"^(.+)\.\d+$", r"\1.gt", subdir.name)
        if subcount != 1:
            print(f"skipping {subdir}")
            continue
        if gt_prefix is None:
            gtdir = subdir.with_name(gtdir_name)
        else:
            gtdir = gt_prefix / gtdir_name

        for imgpath in subdir.iterdir():
            if not imgpath.name.endswith(".png"):
                continue

            component = imgpath.name.removesuffix(".png")
            gt_component = component.rsplit(".", 1)[0]
            result = open_image(imgpath)

            if not (gtdir / f"{gt_component}.png").exists():
                continue

            test = open_image(gtdir / f"{gt_component}.png")

            try:
                (subdir / f"{component}.rms_error").write_text(
                    str(rms_error(test, result))
                )
            except:
                traceback.print_exc()
            try:
                (subdir / f"{component}.rms_lsqw_error").write_text(
                    str(rms_error(test, lsq_whitepoint(test, result)))
                )
            except:
                traceback.print_exc()
            try:
                if lpips_error is not None:
                    (subdir / f"{component}.lpips_error").write_text(
                        str(lpips_error(test, result))
                    )
            except:
                traceback.print_exc()
            try:
                if lpips_error is not None:
                    (subdir / f"{component}.lpips_lsqw_error").write_text(
                        str(lpips_error(test, lsq_whitepoint(test, result)))
                    )
            except:
                traceback.print_exc()


if __name__ == "__main__":
    app()
