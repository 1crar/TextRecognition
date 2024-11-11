import os
import cv2
import time

from dotenv import load_dotenv
# from cv2.dnn_superres import DnnSuperResImpl
from img2table.ocr import TesseractOCR
from img2table.document import Image
from PIL import Image as PILImage

load_dotenv()

# src = "pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png"
#
# # Instantiation of OCR
# #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\melada\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# ocr = TesseractOCR(lang="rus")
#
# # Instantiation of document, either an image or a PDF
# doc = Image(src)
# extracted_tables = doc.extract_tables(ocr=ocr)
#
# table_img = cv2.imread(src)
#
# for table in extracted_tables:
#     for row in table.content.values():
#         for cell in row:
#             cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
#
# print(PILImage.fromarray(table_img).save("pdf_appRecognizer/extract_assets/image_files/temp_2.png"))
# # Table extraction
# extracted_tables = doc.extract_tables(ocr=ocr,
#                                      implicit_rows=True,
#                                      borderless_tables=False,
#                                      min_confidence=50)
#
# doc.to_xlsx(dest='pdf_appRecognizer/extract_assets/xlsx_files/test.xlsx',
#             ocr=ocr)


# def improve_quality(path_to_model: str, path_to_image: str, path_to_save: str) -> None:
#     """
#     А вот эта функция тоже предназначена для улучшения качества изображения. Используется модель EDSR_x3.pb/EDSR_x4.pb
#     Модели были взяты тут: https://github.com/Saafke/EDSR_Tensorflow/tree/master/models
#
#     :param path_to_model: Путь до модели
#     :param path_to_image: Путь до изображения
#     :param path_to_save: Путь для сохранения обработанного изображения через обученную модель.
#     :return: None
#     """
#
#     img = cv2.imread(path_to_image)
#     sr = cv2.dnn_superres.DnnSuperResImpl()
#     sr.readModel(path_to_model)
#
#     sr.setModel("edsr", 4)
#
#     result = sr.upsample(img)
#     cv2.imwrite(img=result, filename=path_to_save)
#
#
# start_time = time.time()
# improve_quality(path_to_model='pdf_appRecognizer/extract_assets/models/EDSR_x4.pb',
#                 path_to_image='pdf_appRecognizer/extract_assets/image_files/Test_YPD_2.png',
#                 path_to_save='pdf_appRecognizer/extract_assets/image_files/temp_2.png')
# end_time = time.time() - start_time
# print(f"время выполнение программы: {end_time} сек.")


from super_image import EdsrModel, DrlnModel, ImageLoader
from PIL import Image

import torch
import numpy as np



# def upscale(img_path: str, upscaled_img: str, model: torch.nn.Module, block_size: int = 256):
#     # В качестве девайса используем видеокарту
#     cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     torch.cuda.empty_cache()
#
#     model = model.to(cur_device)
#
#     img = Image.open(fp=img_path)
#     img = img.convert('RGB')
#
#     width, height = img.size
#
#     # Создаем пустое изображение для сохранения результатов
#     upscaled_img_full = Image.new("RGB", (width * 4, height * 4))
#
#     with torch.no_grad():
#         for i in range(0, height, block_size):
#             for j in range(0, width, block_size):
#
#
#     # Передаем объект PIL непосредственно в ImageLoader.load_image
#     inputs = ImageLoader.load_image(img)
#     preds = model(inputs)
#
#     ImageLoader.save_image(pred=preds, output_file=fr'pdf_appRecognizer/extract_assets/image_files/{upscaled_img}')
#     ImageLoader.save_compare(input=inputs, pred=preds,
#                              output_file=fr'pdf_appRecognizer/extract_assets/image_files/{upscaled_img.replace(
#                                  '.png', '')}_compare.png')


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

    # Сохраняем сравнение
    # ImageLoader.save_compare(input=inputs.cpu(), pred=preds.cpu(),
    #                          output_file=fr'pdf_appRecognizer/extract_assets/image_files/'
    #                                      fr'{upscaled_img.replace(".png", "")}_compare.png')

    # Освобождаем память
    # del inputs
    # del preds
    torch.cuda.empty_cache()


img_path = r'pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2_scale_x4.png'
image = Image.open(fp=img_path)
cur_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

upscale(img_path=img_path, upscaled_img='Test_Table_YPD_2_scale_x4_scale_x2.png', model=cur_model)


# cur_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
# inputs = ImageLoader.load_image(image)
# preds = cur_model(inputs)
#
# ImageLoader.save_image(preds,
#                        r'pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2_scaled_4x.png')                        # save the output 2x scaled image to `./scaled_2x.png`
# ImageLoader.save_compare(inputs, preds,
#                          r'pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2_scaled_4x_compare.png')      # save an output comparing the super-image with a bicubic scaling

