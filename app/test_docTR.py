import torch
import os

from doctr.models import ocr_predictor, crnn_vgg16_bn
from doctr.io import DocumentFile
from doctr.datasets import VOCABS



def test_cropped_extraction(cur_path_folder: str, path_to_crnn_model_name: str, cur_arch: str):
    for cur_cropped in os.listdir(path=cur_path_folder):
        cropped_file_path = os.path.join(cur_path_folder, cur_cropped)
        # Далее идет процесс инициализации crnn модели
        cur_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cur_model = crnn_vgg16_bn(pretrained=False, vocab=VOCABS["ukrainian"])

        params = torch.load(f=path_to_crnn_model_name, map_location=cur_device)
        cur_model.load_state_dict(params)

        cur_predictor = ocr_predictor(det_arch=cur_arch, reco_arch=cur_model, pretrained=True)
        cur_single_img_doct = DocumentFile.from_images(files=[cropped_file_path])

        cur_result = cur_predictor(cur_single_img_doct )
        cur_text_blocks = cur_result.pages[0].blocks

        for cur_text in cur_text_blocks:
            # print(text.lines)
            for cur_el in cur_text.lines:
                # Вывод извлеченного текста с помощью docTR
                print(f'Распознанный текст/элементы текста: {cur_el.words}')

        cur_result.show()


is_test_cropped_files: bool = False
arch_list: list = ['linknet_resnet18', 'db_mobilenet_v3_large']


if is_test_cropped_files:
    path_folder: str = 'pdf_appRecognizer/extract_assets/image_files/YPD_1/cropped_parts'
    local_model_path: str = 'trained_models/docTR/crnn_vgg16_bn_11th_learning.pt'

    test_cropped_extraction(cur_path_folder=path_folder, path_to_crnn_model_name=local_model_path, cur_arch=arch_list[1])
else:
    img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPD_2/YPD_2.png'
    local_model_path: str = 'trained_models/docTR/crnn_vgg16_bn_11th_learning.pt'
    # Подключаем gpu если torch.cuda.is_available() - True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    own_model = crnn_vgg16_bn(pretrained=False, vocab=VOCABS["ukrainian"])
    # based_model = crnn_vgg16_bn(pretrained=True, vocab=)

    model_params = torch.load(f=local_model_path, map_location=device, weights_only=True)
    own_model.load_state_dict(model_params)

    predictor = ocr_predictor(det_arch=arch_list[1], reco_arch=own_model, pretrained=True)

    single_img_doct = DocumentFile.from_images(files=[img_path])

    result = predictor(single_img_doct)
    text_blocks = result.pages[0].blocks

    for text in text_blocks:
        # print(text.lines)
        for el in text.lines:
            # Вывод извлеченного текста с помощью docTR
            print(f'Распознанный текст/элементы текста: {el.words}')

    result.show()
