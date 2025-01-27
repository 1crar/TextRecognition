import torch

from doctr.models import ocr_predictor, crnn_vgg16_bn
from doctr.io import DocumentFile
from doctr.datasets import VOCABS


img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPD_1/UPD_1.png'

# Пути к моделям
local_model_path_1: str = 'trained_models/docTR/crnn_vgg16_bn_20250122-145242_20250123-130213.pt'
local_model_path_2: str = 'trained_models/docTR/crnn_vgg16_bn_20250122-145242_20250123-130213_20250124-122321.pt'

# Подключаем gpu если torch.cuda.is_available() - True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Инициализация первой модели
own_model_1 = crnn_vgg16_bn(pretrained=False, vocab=VOCABS["ukrainian"])
model_params_1 = torch.load(f=local_model_path_1, map_location=device)
own_model_1.load_state_dict(model_params_1)

# Инициализация второй модели
own_model_2 = crnn_vgg16_bn(pretrained=False, vocab=VOCABS["ukrainian"])
model_params_2 = torch.load(f=local_model_path_2, map_location=device)
own_model_2.load_state_dict(model_params_2)

# Инициализация предсказателей для обеих моделей
predictor_1 = ocr_predictor(det_arch='linknet_resnet18', reco_arch=own_model_1, pretrained=True)
predictor_2 = ocr_predictor(det_arch='linknet_resnet18', reco_arch=own_model_2, pretrained=True)

# Загружаем изображение
single_img_doct = DocumentFile.from_images(files=[img_path])

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
