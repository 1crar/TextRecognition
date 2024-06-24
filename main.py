import os
import time
import logging

from dotenv import load_dotenv
from log_config.log_config import activate_logging

from classes.data_json import DataCollection, DataCleaning, DictToJson
from classes.pdf import PdfAnalyzer, PdfTabula, PdfTextReader, PdfCamelot, dt_frame_to_list
from classes.img import ImageDataExtracter, tesseract_languages
from classes.text_extracter import InnInvoiceDataExtraction

# Загружаем переменные из файла .env
load_dotenv()
# Подключаем главный лог
logger = logging.getLogger()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

# Позволяет выбрать из какого формата будем извлекать (если True, то из картинки. False - из пдф)
IS_IMAGE: bool = False

if __name__ == '__main__':
    # Write list of available languages to file
    with open(file='tesseract_languages.txt', mode='w', encoding='utf-8') as f:
        print(tesseract_languages(path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}'), file=f)

    if IS_IMAGE:
        activate_logging()
        start = time.time()
        image = ImageDataExtracter(path_dir='extract_assets/input_files', image_file='IMG_20240603_191413.jpg',
                                   path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}', language='rus')
        result = image.extract_data_from_image()
        print(result, '---------------Execution time---------------', f'{(time.time() - start):.2f}', sep='\n')

        # записываем результат в extracted_results/Image_result.txt
        with open(file='extracted_results/Image_result.txt', mode='w', encoding='utf-8') as f:
            print(result, file=f)
    else:
        start = time.time()
        activate_logging()

        # Создаем экземпляр класса на основе либы камелот (для извлечения табличной части из pdf)
        camelot_instance = PdfCamelot(path_dir='extract_assets/input_files/upds_and_invoices',
                                      pdf_file='Универсальный передаточный документ (УПД) с факсимилье № ЦБ-460 от 31.05.2024.pdf')
        # Считываем таблицы с помощью камелота
        tables = camelot_instance.read_tables()
        # Получаем кол-во таблиц для обработки
        table_numbers: int = tables.n
        # Далее соединяем в одну единую таблицу, если camelot выделил больше одной таблицы (table_numbers > 1)
        if table_numbers == 1:
            table: list = tables[0].data
        else:
            table: list = []
            for i in range(0, table_numbers):
                table += tables[i].data
        # Затем импортируем класс DataCleaning для очистки и работы с извлеченной таблицей
        cleaned_table: list = DataCleaning.data_clean(data_table=table)
        # Далее извлекаем текст из pdf (без учета структуры) для извлечения данных вне табличной части
        # (ИНН/КПП, Счет-фактура)
        pdf = PdfTextReader(path_dir='extract_assets/input_files/upds_and_invoices',
                            pdf_file='Универсальный передаточный документ (УПД) с факсимилье № ЦБ-460 от 31.05.2024.pdf')
        # Создаем экземпляр класса текст
        text = pdf.extract_text_from_pdf()
        # Извлекаем текст из pdf
        my_regulars: 'InnInvoiceDataExtraction' = InnInvoiceDataExtraction(text=text)
        logger.info('Извлеченный текст документа:\n\n%s\n', my_regulars.text)
        # Извлекаем остальные данные (ИНН/КПП, Счет-фактура, Итоговая сумма)
        inn_kpp: str = my_regulars.inn_and_kpp_extract()
        invoice: str = my_regulars.invoice_extract()
        total: str = my_regulars.total_sum_extract(data_table=cleaned_table)
        # contract_number: str = my_regulars.contract_extract()

        # Формируем хэш-таблицу на основе полученных данных
        data = DataCollection()
        collection = data.data_collect(inn_kpp=inn_kpp, invoice=invoice, cleaned_data=cleaned_table, total=total)
        if type(collection) == dict:
            # Записываем итоговую хэш-таблицу в json файл
            DictToJson.write_to_json(collection=collection)
        logger.info('---------------Execution time: %s---------------', f'{(time.time() - start):.2f} seconds')

