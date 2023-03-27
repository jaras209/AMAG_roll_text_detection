from pathlib import Path
from tqdm import tqdm

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

directory = Path("data/rolls")
output_dir = Path('data/craft_outputs_2')
directory_iter = sorted(directory.iterdir())

# load models
refine_net = load_refinenet_model(cuda=False)
craft_net = load_craftnet_model(cuda=False)

for image_path in tqdm(directory_iter, total=len(directory_iter)):
    if image_path.name == '.DS_Store':
        continue

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
        continue

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

# unload models from gpu
empty_cuda_cache()
