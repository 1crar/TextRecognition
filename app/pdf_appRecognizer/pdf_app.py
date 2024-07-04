import os
import time
import logging

from dotenv import load_dotenv
from .log_config.log_config import activate_logging

from .classes.data_json import DataCollection, DataCleaning, DictToJson
from .classes.pdf import PdfTextReader, PdfCamelot
from .classes.text_extracter import DataExtraction

# Загружаем переменные из файла .env
load_dotenv()
# Подключаем главный лог
logger = logging.getLogger()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')


def pdf_app(pdf_filename: str):
    start = time.time()
    activate_logging()
    # Создаем экземпляр класса на основе либы camelot (для извлечения табличной части из pdf)
    camelot_instance = PdfCamelot(path_dir='extract_assets/uploaded_files',
                                  pdf_file=pdf_filename)
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
    pdf = PdfTextReader(path_dir='extract_assets/uploaded_files',
                        pdf_file=pdf_filename)
    # Создаем экземпляр класса текст
    text = pdf.extract_text_from_pdf()
    # Извлекаем текст из pdf
    my_regulars: 'DataExtraction' = DataExtraction(text=text)
    logger.info('Извлеченный текст документа:\n\n%s\n', my_regulars.text)
    # Извлекаем остальные данные (ИНН/КПП, Счет-фактура)
    inn_kpp: str = my_regulars.inn_and_kpp_extract()
    invoice: str = my_regulars.invoice_extract()
    # Извлекаем от
    totals: tuple = my_regulars.total_sum_extract(data_table=cleaned_table)
    # Формируем хэш-таблицу на основе полученных данных
    data = DataCollection()
    collection = data.data_collect(inn_kpp=inn_kpp, invoice=invoice, cleaned_data=cleaned_table, totals=totals)
    logger.info('---------------Execution time: %s---------------', f'{(time.time() - start):.2f} seconds')
    return collection


# if __name__ == '__main__':
#     start = time.time()
#     activate_logging()
#     # Создаем экземпляр класса на основе либы camelot (для извлечения табличной части из pdf)
#     camelot_instance = PdfCamelot(path_dir='extract_assets/input_files/upds_and_invoices',
#                                   pdf_file='УПД 31.05.24 № 428 = 257 428.00 без НДС.pdf')
#     # Считываем таблицы с помощью камелота
#     tables = camelot_instance.read_tables()
#     # Получаем кол-во таблиц для обработки
#     table_numbers: int = tables.n
#     # Далее соединяем в одну единую таблицу, если camelot выделил больше одной таблицы (table_numbers > 1)
#     if table_numbers == 1:
#         table: list = tables[0].data
#     else:
#         table: list = []
#         for i in range(0, table_numbers):
#             table += tables[i].data
#     # Затем импортируем класс DataCleaning для очистки и работы с извлеченной таблицей
#     cleaned_table: list = DataCleaning.data_clean(data_table=table)
#     # Далее извлекаем текст из pdf (без учета структуры) для извлечения данных вне табличной части
#     # (ИНН/КПП, Счет-фактура)
#     pdf = PdfTextReader(path_dir='extract_assets/input_files/upds_and_invoices',
#                         pdf_file='УПД 31.05.24 № 428 = 257 428.00 без НДС.pdf')
#     # Создаем экземпляр класса текст
#     text = pdf.extract_text_from_pdf()
#     # Извлекаем текст из pdf
#     my_regulars: 'DataExtraction' = DataExtraction(text=text)
#     logger.info('Извлеченный текст документа:\n\n%s\n', my_regulars.text)
#     # Извлекаем остальные данные (ИНН/КПП, Счет-фактура)
#     inn_kpp: str = my_regulars.inn_and_kpp_extract()
#     invoice: str = my_regulars.invoice_extract()
#     # Извлекаем от
#     totals: tuple = my_regulars.total_sum_extract(data_table=cleaned_table)
#     # Формируем хэш-таблицу на основе полученных данных
#     data = DataCollection()
#     collection = data.data_collect(inn_kpp=inn_kpp, invoice=invoice, cleaned_data=cleaned_table, totals=totals)
#     logger.info('---------------Execution time: %s---------------', f'{(time.time() - start):.2f} seconds')

