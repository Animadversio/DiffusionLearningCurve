import os
import glob
import re
import json
from typing import Dict, List, Tuple

import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw

# common pre-processing

def preprocess_image(img: Image.Image, scale: int = 3) -> Image.Image:
    """
    Convert image to grayscale and upscale.
    """
    gray = img.convert('L')
    return gray.resize((gray.width * scale, gray.height * scale), Image.BICUBIC)



def get_montage_patches_pil(
    img, 
    patch_size=32, 
    pad=4, 
    margin=None
):
    """
    Args:
      img:         PIL.Image or path to image
      patch_size:  size of each square tile
      pad:         pixels between tiles
      margin:      pixels of padding around the full montage;
                   if None, assumed equal to `pad`
    Returns:
      List[PIL.Image] of exactly patch_size×patch_size tiles.
    """
    if isinstance(img, str):
        img = Image.open(img)
    W, H = img.size
    if margin is None:
        margin = pad

    # how many columns/rows of tiles?
    n_cols = (W - 2*margin + pad) // (patch_size + pad)
    n_rows = (H - 2*margin + pad) // (patch_size + pad)

    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = margin + j*(patch_size + pad)
            y = margin + i*(patch_size + pad)
            box = (x, y, x + patch_size, y + patch_size)
            patches.append(img.crop(box))
    return patches


def ocr_image(img: Image.Image, config: str) -> Tuple[str, Dict]:
    """
    Run Tesseract OCR on the image.
    Returns raw text and structured data dict.
    """
    raw_text = pytesseract.image_to_string(img, config=config)
    data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    return raw_text, data


def parse_ocr_data(data: Dict, min_confidence: int = 30) -> Dict[int, List[Tuple[str, int, Tuple[int,int,int,int]]]]:
    """
    Group OCR data into lines, filtering by confidence.
    Returns a dict: line_num -> list of (word, confidence, bbox).
    """
    lines: Dict[int, List[Tuple[str, int, Tuple[int,int,int,int]]]] = {}
    n = len(data.get('text', []))
    for i in range(n):
        word = data['text'][i].strip()
        if not word:
            continue
        conf = int(data['conf'][i])
        if conf < min_confidence:
            continue
        ln = data['line_num'][i]
        bbox = (
            data['left'][i], data['top'][i],
            data['width'][i], data['height'][i]
        )
        lines.setdefault(ln, []).append((word, conf, bbox))
    return lines


def draw_boxes(img: Image.Image, lines: Dict[int, List[Tuple[str,int,Tuple[int,int,int,int]]]],
               outline: str = 'red', text_fill: str = 'red') -> Image.Image:
    """
    Draw bounding boxes and confidence labels on the image.
    Returns a new image with drawings.
    """
    draw = ImageDraw.Draw(img)
    for ln, words in lines.items():
        for w, conf, (x, y, wdt, hgt) in words:
            draw.rectangle([x, y, x + wdt, y + hgt], outline=outline, width=1)
            draw.text((x, y - 10), f"{w}({conf})", fill=text_fill)
    return img


def process_step_image(img_path: str, config: str, output_dir: str) -> Dict:
    """
    Process a single montage image: extract patches, run OCR, draw boxes, save image and return structured results.
    """

    step = int(re.search(r'samples_epoch_(\d+)\.png', os.path.basename(img_path)).group(1))
    results = {
        'step': step,
        'raw_text': [],  # list of raw OCR dumps per patch
        'lines': {}      # mapping patch_idx -> parsed lines
    }

    patches = get_montage_patches_pil(img_path, patch_size=32, pad=1)
    for idx, patch in enumerate(patches):
        img = preprocess_image(patch)
        raw_text, data = ocr_image(img, config)
        parsed = parse_ocr_data(data)

        results['raw_text'].append({'patch_idx': idx, 'text': raw_text})
        results['lines'][idx] = parsed
        if (idx == 0):
            # draw and save per-patch if desired, or draw on montage
            drawn = draw_boxes(img.copy(), parsed)
            drawn_path = os.path.join(output_dir, f"step_{step:06d}_patch_{idx:03d}.png")
            drawn.save(drawn_path)

    # save JSON result for this step
    json_path = os.path.join(output_dir, f"step_{step:06d}_ocr.json")
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=2)

    return results


def main(imgdir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sample_files = sorted(
        glob.glob(os.path.join(imgdir, "samples_epoch_*.png"))
    )

    BASE_CONFIG = (
        '--oem 1 --psm 6 '
        '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        '-c load_system_dawg=0 -c load_freq_dawg=0 -c wordlist_file='
    )

    all_results = []
    for imgpath in sample_files:
        print(f"Processing {imgpath}...")
        res = process_step_image(imgpath, BASE_CONFIG, output_dir)
        all_results.append(res)

    # Optionally save aggregated results
    agg_path = os.path.join(output_dir, "all_steps_ocr_results.json")
    with open(agg_path, 'w') as af:
        json.dump(all_results, af, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="OCR montage patches and save results.")
    parser.add_argument('--imgdir', required=True, help="Directory containing sample images.")
    parser.add_argument('--out', required=True, help="Output directory for JSON and drawn images.")
    args = parser.parse_args()

    main(args.imgdir, args.out)

# # %% [markdown]
# # ### Try to PytesseractOCR the generated text

# # %%
# import os
# from PIL import Image
# import pytesseract

# # Example:
# # patches = get_montage_patches_pil("montage.png", patch_size=32, pad=4)
# # print(len(patches), patches[0].size)  # -> (32, 32)

# # %%
# # If Tesseract isn't in your PATH, specify the full path:
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# imgdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/words32x32_50k_UNet_CNN_EDM_4blocks_noattn/samples"
# imgpath = os.path.join(imgdir, "samples_epoch_050000.png")
# # Load an image
# img = Image.open(imgpath)
# # Perform OCR
# text = pytesseract.image_to_string(img)
# print(text)

# # %%
# # %%
# import os
# from PIL import Image
# import pytesseract

# # If Tesseract isn't in your PATH, specify the full path:
# # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# # imgdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/words32x32_50k_UNet_CNN_EDM_4blocks_noattn/samples"
# # imgpath = os.path.join(imgdir, "samples_epoch_010000.png")

# # # %%
# # patches = get_montage_patches_pil(imgpath, patch_size=32, pad=1)
# # for patch in patches:
# #     print(pytesseract.image_to_string(patch))
# # %% [markdown]
# # ### Tesseeract Configured version

# # %%
# import glob
# import re
# import pytesseract
# from pytesseract import Output
# from PIL import Image, ImageDraw

# # common pre‐processing
# def prep(img):
#     img = img.convert('L')                                    # gray
#     img = img.resize((img.width*3, img.height*3), Image.BICUBIC)  
#     return img

# # config: LSTM only, block of text, no dictionaries, whitelist
# BASE_CONFIG = (
#     '--oem 1 '
#     '--psm 6 '                       # assume a uniform block of text
#     '-c tessedit_char_whitelist='
#       'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#       'abcdefghijklmnopqrstuvwxyz'
#     #   '0123456789 '
#     ' '
#     '-c load_system_dawg=0 '
#     '-c load_freq_dawg=0 '
#     '-c wordlist_file='
# )

# imgdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/words32x32_50k_UNet_CNN_EDM_4blocks_noattn/samples"
# imgpath = os.path.join(imgdir, "samples_epoch_010000.png")
# sample_files = glob.glob(os.path.join(imgdir, "samples_epoch_*.png"))
# steps = [int(re.search(r'samples_epoch_(\d+)\.png', os.path.basename(f)).group(1)) for f in sample_files]

# for step in steps:
#     imgpath = os.path.join(imgdir, f"samples_epoch_{step:06d}.png")
#     patches = get_montage_patches_pil(imgpath, patch_size=32, pad=1)
#     for patch in patches:
#         img = prep(patch)
#         # 1) full‐text dump (keeps line breaks)
#         txt = pytesseract.image_to_string(img, config=BASE_CONFIG)
#         print("RAW TEXT:\n", txt)
#         # 2) structured data if you want bounding boxes + confidences
#         data = pytesseract.image_to_data(
#             img,
#             config=BASE_CONFIG,
#             output_type=Output.DICT
#         )
        
#         # reconstruct lines
#         lines = {}
#         n = len(data['text'])
#         for i in range(n):
#             word = data['text'][i].strip()
#             if not word: 
#                 continue
#             ln = data['line_num'][i]
#             conf = int(data['conf'][i])
#             # filter out very low confidence
#             if conf < 30:
#                 continue
#             lines.setdefault(ln, []).append((word, conf,
#                                             (data['left'][i],
#                                             data['top'][i],
#                                             data['width'][i],
#                                             data['height'][i])))
        
#         # print per‐line
#         for ln in sorted(lines):
#             words = [w for w, _, _ in lines[ln]]
#             print(f"Line {ln:02d}: {' '.join(words)}")
        
#         # (optional) draw boxes
#         draw = ImageDraw.Draw(img)
#         for ln in lines:
#             for w, conf, (x, y, wid, ht) in lines[ln]:
#                 draw.rectangle([x, y, x+wid, y+ht], outline='red', width=1)
#                 draw.text((x, y-10), f"{w}({conf})", fill='red')
#         # display(img)

# %%
# imgdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/words32x32_50k_UNet_CNN_EDM_4blocks_noattn/samples"
# step = 50000
# imgpath = os.path.join(imgdir, f"samples_epoch_{step:06d}.png")
# patches = get_montage_patches_pil(imgpath, patch_size=32, pad=1)
# for patch in patches:
#     print(pytesseract.image_to_string(patch))
#     display(patch)

# # %%
# import os
# import glob
# import re
# from tqdm.auto import tqdm

# imgdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiffusionSpectralLearningCurve/words32x32_50k_UNet_CNN_EDM_4blocks_noattn/samples"

# # Find all sample files in the directory
# sample_files = glob.glob(os.path.join(imgdir, "samples_epoch_*.png"))
# steps = [int(re.search(r'samples_epoch_(\d+)\.png', os.path.basename(f)).group(1)) for f in sample_files]
# steps.sort()

# # Dictionary to store results: step -> list of recognized texts
# ocr_results = {}

# for step in tqdm(steps): # 1hr + for 500 * 64 samples 
#     imgpath = os.path.join(imgdir, f"samples_epoch_{step:06d}.png")
#     patches = get_montage_patches_pil(imgpath, patch_size=32, pad=1)
    
#     # Extract text from each patch
#     texts = []
#     for patch in patches:
#         text = pytesseract.image_to_string(patch).strip()
#         texts.append(text)
    
#     # Store in dictionary
#     ocr_results[step] = texts
    
#     # Print a sample to see progress
#     print(f"Step {step}: Found {len(texts)} text samples")
#     print(f"Sample texts: {texts[:3]}")

