import cv2

from dotenv import load_dotenv
from img2table.document import Image
from PIL import Image, ImageDraw, ImageFont

import torch
import easyocr
import numpy as np

from img2table.ocr import EasyOCR
from PIL import Image as Img

load_dotenv()


def upscale(img_path: str, upscaled_img: str, model: torch.nn.Module):
    cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(cur_device)

    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    img = img[:, :, 0:3]

    tileCountX = 16
    tileCountY = 16

    M = img.shape[0] // tileCountX
    N = img.shape[1] // tileCountY

    tiles = [[img[x:x + M, y:y + N] for x in range(0, img.shape[0], M)] for y in range(0, img.shape[1], N)]
    inputs = [[torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(cur_device) for tile in part] for part in tiles]

    upscaled = None
    count = 0

    for i in range(tileCountY + 1):
        col = None
        for j in range(tileCountX + 1):
            pred = model(inputs[i][j])
            res = pred.detach().to('cpu').squeeze(0).permute(1, 2, 0)
            # print(f"Image tile #{count}. Upscaled shape: {res.shape}")
            count += 1
            col = res if col is None else torch.cat([col, res], dim=0)
            del pred
        upscaled = col if upscaled is None else torch.cat([upscaled, col], dim=1)

    # Сохраняем итоговое изображение
    cv2.imwrite(fr'pdf_appRecognizer/extract_assets/image_files/{upscaled_img}',
                upscaled.numpy() * 255.0)

    torch.cuda.empty_cache()


enhanced_file: str = 'temp/scaled_4_enhanced_test_2.png'
image_path: str = f'pdf_appRecognizer/extract_assets/image_files/dt_cells/{enhanced_file}'

image_cv2 = cv2.imread(filename=enhanced_file)
languages: list = ['ru']

# dt_img2excel(img_path=image_path)


print("[INFO] OCR'ing input image...")
print(f"[INFO] OCR'ing with the following languages: {languages}")

test_case: int = 2

if test_case == 1:
    reader = easyocr.Reader(lang_list=languages, gpu=True,
                            model_storage_directory='fine_tune', user_network_directory='fine_tune',
                            recog_network='ru_finetune')
    # results = reader.readtext(image_cv2, text_threshold=0.75, contrast_ths=0.7, width_ths=1.25, height_ths=0.75,
    #                           ycenter_ths=0.5, slope_ths=1, add_margin=0.175, decoder='wordbeamsearch', beamWidth=20,
    #                           canvas_size=3500)
    # Not bad
    results = reader.readtext(image_cv2, text_threshold=0.75, contrast_ths=0.8, width_ths=1.3, height_ths=0.75,
                              ycenter_ths=0.5, slope_ths=1, add_margin=0.200, decoder='wordbeamsearch', beamWidth=20,
                              canvas_size=3500)
if test_case == 2:
    reader = easyocr.Reader(lang_list=languages, gpu=True)
    # results = reader.readtext(image_cv2)
    results = reader.readtext(image_cv2)

# Преобразуем изображение OpenCV в формат PIL
image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)

# Выберите шрифт (если нужный шрифт не установлен, вы можете указать путь к .ttf файлу)
font = ImageFont.truetype(font="arial.ttf", size=20)  # Убедитесь, что путь к шрифту корректный

for (bbox, text, prob) in results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    # Распаковка координат
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))

    # Рисуем прямоугольник
    draw.rectangle([tl, br], outline=(0, 255, 0), width=2)

    # Рисуем текст
    draw.text((tl[0], tl[1] - 10), text, fill=(0, 255, 0), font=font)

# Отображаем результат
image_pil.show()

# Сохранение изображения (необязательно)
# image_pil.save('test_3.png')


# reader = easyocr.Reader(['ru'])
#
# img = Image.open(fp='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhance_ver2_UPD_1_scaled_3.png')
#
# gray_img = img.convert('L')
# gray_img = cv2.imread('pdf_appRecognizer/extract_assets/image_files/YPD_1/enhance_ver2_UPD_1_scaled_3.png')
#
# result = reader.readtext(gray_img, detail=0)

# extracted_data: dict = {
#     'extracted_data': result
# }
#
# with open(file='pdf_appRecognizer/extract_assets/json_files/extracted_data_easyOCR.json',
#           mode='w', encoding='utf-8') as json_file:
#     json.dump(obj=extracted_data, fp=json_file, ensure_ascii=False, indent=3)
#
# print(result)


# text_to_image(text="Привет, мир!", font_size=40, output_filename='1.png')
# dirty_image(img='pdf_appRecognizer/extract_assets/image_files/generated_assets/0.png', mode='b')
