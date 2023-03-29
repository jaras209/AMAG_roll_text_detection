from pathlib import Path
from tqdm import tqdm
import xlsxwriter

from extract_text import extract_text_from_roll

if __name__ == '__main__':
    directory = Path("data/rolls")
    output_dir = Path('data/craft_outputs')
    directory_iter = sorted(directory.iterdir())

    workbook = xlsxwriter.Workbook(
        output_dir / "extracted_text.xlsx"
    )
    worksheet = workbook.add_worksheet()
    text_format = workbook.add_format({"num_format": "@"})

    row_num = 1
    for image_path in tqdm(directory_iter, total=len(directory_iter)):
        if image_path.name == '.DS_Store':
            continue

        crops_paths, extracted_texts = extract_text_from_roll(image_path, output_dir)

        for crop_path, text in zip(crops_paths, extracted_texts):
            worksheet.write(f"A{row_num}", str(image_path), text_format)
            worksheet.write(f"B{row_num}", str(crop_path), text_format)
            worksheet.write(f"C{row_num}", str(text), text_format)
            worksheet.insert_image(f"D{row_num}", crop_path)
            row_num += 1

    workbook.close()
