from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

directory = Path("data/roll_crops")
for image_path in sorted(directory.iterdir()):
    if image_path.name == '.DS_Store':
        continue

    print(f"Working on {image_path.name}...")
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"\t Extracted text:\n\t {extracted_text}")
    print(f"{'=' * 10}")
