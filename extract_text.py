from pathlib import Path
from typing import Union
from PIL import Image

from tqdm import tqdm
import torch

# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

CUDA = torch.cuda.is_available()

# load models
refine_net = load_refinenet_model(cuda=CUDA)
craft_net = load_craftnet_model(cuda=CUDA)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")


def extract_text_from_roll(image_path: Union[str, Path], output_dir: Union[str, Path]):
    image_path = Path(image_path)
    extracted_texts = []

    # read image
    image = read_image(str(image_path))

    # perform prediction
    try:
        prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=1280
        )
    except ValueError as er:
        print(f"Error while extracting image {image_path}: {er}")
        return [], []

    # export detected text regions
    exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        output_dir=str(output_dir / image_path.name),
        rectify=True
    )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=str(output_dir / image_path.name)
    )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        output_dir=str(output_dir / image_path.name)
    )
    for crop_path in exported_file_paths:
        crop = Image.open(crop_path).convert("RGB")
        pixel_values = processor(crop, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)

        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        extracted_texts.append(extracted_text)
        print(f"\t Extracted text:\n\t {extracted_text}")
        print(f"{'=' * 10}")

    return exported_file_paths, extracted_texts
