import os
import re

from dotenv import load_dotenv
from classes.img import Img
from classes.text_extracter import PatternDataExtraction
from classes.pdf import PDF, extract_tables


load_dotenv()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

if __name__ == '__main__':
    # print(f'{path_to_tesseract}\\{tesseract_exe}')

    pdf_example = PDF(path_dir='extract_assets', pdf_file='BL24-union.pdf')

    if pdf_example.is_contains_data_table():

        pdf_table = extract_tables(path_to_pdf=pdf_example.full_path)

        dt = []  # список двумерных массивов (двумерных списков)
        for el in pdf_table:
            if type(el) == list:
                dt.append(el)

        for i in range(len(pdf_table)):
            dt.append(pdf_table[i][1])

        # print(dt, len(dt), dt[0][0], sep='\n')

        # for i in range(len(dt)):
        #     for j in range(len(dt[i])):
        #         print(dt[i][j])

        # Записываем извлеченный данные из таблицы в текстовый файл
        with open(file='data.txt', encoding='utf-8', mode='w') as data_file:
            for i in range(len(dt)):
                for j in range(len(dt[i])):
                    print(*dt[i][j], file=data_file)

    # Извлекаем сплошным текстом*
    with open(file='data.txt', encoding='utf-8', mode='r') as data_file:
        data = data_file.readlines()

    text = ''
    for el in data:
        text += el

    data_extraction = PatternDataExtraction(txt=text)
    article = data_extraction.extract_article_number()
    quantity = data_extraction.extract_quantity()
    # print(article, quantity, sep='\n')

    articles = re.finditer(r"(Z\d{6})", text)
    for article in articles:
        print(article.group(0))

    # Извлекаем текст из пдф-файла
    # image_1 = Img(path_dir=path_to_tesseract, exe_file=tesseract_exe)
    # extracted_text = image_1.get_text(image=r'extract_assets/BL24-10003-1.png')
    #
    # # Извлекаем номер артикула из текста (который был извлечен из пдф)
    # data_extraction = PatternDataExtraction(txt=extracted_text)
    # our_data = data_extraction.extract_article_number()
    # our_data = data_extraction.extract_quantity()
    # print(our_data)






