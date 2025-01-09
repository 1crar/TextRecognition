import torch

from doctr.models import ocr_predictor, crnn_vgg16_bn
from doctr.io import DocumentFile
from doctr.datasets import VOCABS


img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPD_1/upscaled_4_UPD_1_dt_part.png'
local_model_path: str = 'trained_models/docTR/crnn_vgg16_bn_20241226-154828.pt'

own_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["ukrainian"])
model_params = torch.load(f=local_model_path, map_location="cpu")
own_model.load_state_dict(model_params)

predictor = ocr_predictor(det_arch='linknet_resnet18', reco_arch=own_model, pretrained=True)

single_img_doct = DocumentFile.from_images(files=[img_path])

result = predictor(single_img_doct)
text_blocks = result.pages[0].blocks

for text in text_blocks:
    # print(text.lines)
    for el in text.lines:
        # Вывод извлеченного текста с помощью docTR
        print(f'Распознанный текст: {el.words}')


result.show()

