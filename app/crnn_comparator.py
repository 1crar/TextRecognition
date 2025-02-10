import torch

from doctr.models import ocr_predictor, crnn_vgg16_bn
from doctr.io import DocumentFile
from doctr.datasets import VOCABS


def model_comparator(path_to_img: str, path_to_model_1: str, path_to_model_2: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Инициализация первой модели
    model_1 = crnn_vgg16_bn(pretrained=False, vocab=VOCABS["ukrainian"])
    model_params_1 = torch.load(f=path_to_model_1, map_location=device)
    model_1.load_state_dict(model_params_1)

    # Инициализация второй модели
    model_2 = crnn_vgg16_bn(pretrained=False, vocab=VOCABS["ukrainian"])
    model_params_2 = torch.load(f=path_to_model_2, map_location=device)
    model_2.load_state_dict(model_params_2)

    # Инициализация предсказателей для обеих моделей
    predictor_1 = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch=model_1, pretrained=True)
    predictor_2 = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch=model_2, pretrained=True)

    # Загружаем изображение
    single_img_doct = DocumentFile.from_images(files=[path_to_img])

    # Получаем результаты от первой модели
    result_1 = predictor_1(single_img_doct)
    text_blocks_1 = result_1.pages[0].blocks

    print("Результаты первой модели:")
    for text in text_blocks_1:
        for el in text.lines:
            print(f'Распознанный текст/элементы текста: {el.words}')

    # Получаем результаты от второй модели
    result_2 = predictor_2(single_img_doct)
    text_blocks_2 = result_2.pages[0].blocks

    print("Результаты второй модели:")
    for text in text_blocks_2:
        for el in text.lines:
            print(f'Распознанный текст/элементы текста: {el.words}')

    # Отображение результатов
    result_1.show()
    result_2.show()


img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPD_1/UPD_1.png'

# Пути к моделям
local_model_path_1: str = 'trained_models/docTR/crnn_vgg16_bn_10th_learning.pt'
local_model_path_2: str = 'trained_models/docTR/crnn_vgg16_bn_11th_learning.pt'


model_comparator(path_to_img=img_path,
                 path_to_model_1=local_model_path_1,
                 path_to_model_2=local_model_path_2)
