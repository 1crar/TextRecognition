import os

from dotenv import load_dotenv
from classes.pdf import PdfAnalyzer, extract_tables
from classes.img import ImageDataTableExtracter, tesseract_languages

# Загружаем переменные из файла .env
load_dotenv()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

# Позволяет выбрать из какого формата будем извлекать (если True, то из картинки. False - из пдф)
IS_IMAGE: bool = True

if __name__ == '__main__':
    # List of available languages
    # print(tesseract_languages(path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}'))
    if IS_IMAGE:
        image_1 = ImageDataTableExtracter(path_dir='extract_assets/input_files', image_file='img_3.png',
                                          path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}')
        result = image_1.extract_data_from_image()
        print(result)

        # записываем результат в extracted_results/img_1.txt
        with open(file='extracted_results/img_1.txt', mode='w', encoding='utf-8') as f:
            print(result, file=f)
    else:
        # Создаем экземпляр класса и присваиваем конкретный pdf-документ, который будем парсить
        pdf_example = PdfAnalyzer(path_dir='extract_assets', pdf_file='pdf_1.pdf')
        print(pdf_example.full_path)

        # Условие срабатывает, если в pdf-документе есть табличная часть
        if pdf_example.is_contains_data_table():
            # Функция возвращает данные в виде списка, извлеченные из табличной части
            pdf_table = extract_tables(path_to_pdf=pdf_example.full_path)
            print(f'данные в pdf_table\n{pdf_table}')






        # Далее работаем со списком pdf_table, так как он очень громоздкий (в этом списке кортеж,
        #     # в котором также содержится список (внутри этого списка, списки самих данных))
        #
        #     dt = []  # Список двумерных массивов (двумерных списков). По сути, сама табличная часть (таблица из pdf-файла)
        #
        #     for el in pdf_table:
        #         # Конкретно извлекаем список
        #         if type(el) == list:
        #             dt.append(el)
        #     # Из списка извлекаем списки с табличными данными
        #     for i in range(len(pdf_table)):
        #         dt.append(pdf_table[i][1])
        #
        #     # print(dt, len(dt), dt[0][0], sep='\n')
        #
        #     # Записываем извлеченный данные из таблицы (dt) в текстовый файл
        #     # Запись текст я реализовал для удобства просмотра результата извлеченных данных.
        #     with open(file='data.txt', encoding='utf-8', mode='w') as data_file:
        #         for i in range(len(dt)):
        #             for j in range(len(dt[i])):
        #                 print(*dt[i][j], file=data_file)
        #
        # # Извлекаем сплошным текстом*
        # with open(file='data.txt', encoding='utf-8', mode='r') as data_file:
        #     data = data_file.readlines()
        #
        # # Создаем var text для записи данных в виде текста
        # text: str = ''
        # for el in data:
        #     text += el
        #
        # # Затем создаем экземпляр класса data_extraction
        # data_extraction = PatternDataExtraction(txt=text)
        #
        # # Создаем список articles и добавляем туда извлеченный данные с помощью метода extract_article_number()
        # articles: list = data_extraction.extract_article_number()
        # # Создаем список quantities и добавляем туда извлеченный данные с помощью метода extract_quantity()
        # quantities: list = data_extraction.extract_quantity()
        # # В созданный ранее словарь добавляем Liefer-Termin
        # terms: list = data_extraction.extract_term_delivery()
        # # Создаем словарь и добавляем в него списки articles, quantities и terms
        # data_collection: dict = data_extraction.data_collect(articles=articles, quantities=quantities,
        #                                                      terms_delivery=terms)
        # print(articles, quantities, terms, data_collection, sep='\n')