import re
import json
from logging import getLogger

logger = getLogger(__name__)        # __name__ - имя модуля


class PatternDataExtraction:
    def __init__(self, txt: str):
        self.txt: str = txt

        self.article_numbers: list = []
        self.quantity_numbers: list = []
        self.term_delivery_numbers: list = []

        self.data_collection: dict = {}

    def extract_article_number(self) -> list:
        article_pattern = r"(Z\d{6})"       # Регулярка для извлечения номера артикула
        result = re.finditer(article_pattern, self.txt)

        for article in result:
            self.article_numbers.append(article.group(0))

        # self.extracted_data['article_number'] = articles
        return self.article_numbers

    def extract_quantity(self) -> list:
        quantity_pattern = r'\d{1,9},\d{2,3} KW'    # Регулярка для извлечения кол-ва (menge)
        result = re.finditer(quantity_pattern, self.txt, re.DOTALL)

        for quantity in result:
            self.quantity_numbers.append(quantity.group(0)[:2])

        # self.extracted_data['quantity'] = quantity_values
        return self.quantity_numbers

    def extract_term_delivery(self) -> list:
        term_delivery_pattern = r'KW\s+(\d+)'      # Регулярка для извлечения Liefer-Termin
        result = re.finditer(term_delivery_pattern, self.txt, re.DOTALL)

        for term in result:
            self.term_delivery_numbers.append(term.group(0)[3:])

        # self.extracted_data['term_delivery'] = term_delivery_values
        return self.term_delivery_numbers

    def data_collect(self, articles: list, quantities: list, terms_delivery: list) -> dict:
        self.data_collection['article_number'] = articles
        self.data_collection['quantity'] = quantities
        self.data_collection['term_delivery'] = terms_delivery

        return self.data_collection


class InnInvoiceDataExtraction:
    def __init__(self, text: str):
        self._text = text

        self.inn_kpp: str = ''
        self.invoice: str = ''
        self.contract_number: str = ''
        self.data_collection: dict = {}

    @property
    def text(self) -> str:
        return self._text

    def inn_and_kpp_extract(self) -> str:
        # pattern_inn_kpp = r'\d{10}/\d{9}'
        pattern_inn_kpp = r'\d{12}|(\d{10}(?:\/\d{9})?)'
        result = re.search(pattern_inn_kpp, self._text.replace(' ', '').lower())
        try:
            self.inn_kpp = result.group(0)
        except AttributeError as e:
            logger.error('Ошибка в атрибуте self.inn_kpp: %s', e)

        logger.info("ИНН/КПП %s", self.inn_kpp)

        return self.inn_kpp
    
    def invoice_extract(self) -> str:
        # Регулярное выражение для извлечения номера Счет-фактуры
        # pattern_invoice_number = re.compile(r'Счет-фактура№(\d{7}/\d{4}) от (\d{2}.\d{2}.\d{4})')
        # pattern_invoice_number = re.compile(r'Cчет-фактура № ([^ ]+)')   # '\d+'
        pattern_invoice_number = re.compile(r'счет-фактура№[^\s]+')
        result = re.search(pattern_invoice_number, self._text.replace(' ', '').lower())
        try:
            self.invoice = result.group(0)
        except AttributeError as e:
            logger.error('Ошибка в атрибуте self.invoice: %s', e)

        logger.info("счет-фактура документа: %s", self.invoice)
        return self.invoice

    def contract_extract(self) -> str:
        matches = re.findall(r'Договор №(\w+)\s+от\s+([\d.]+)', self._text)
        new_matches = re.findall(r'Основание|Основания передачи (сдачи) / получения (приемки) \W+\d+|\S+',
                                 self._text)
        logger.info('Совпадение контрактов\n%s', new_matches)

        for match in new_matches:
            self.contract_number = f'№{match[0]} от {match[1]}'
        return self.contract_number

    def data_collect(self, inn_kpp: str, invoice: str, contract_number: str, data_table: list) -> dict | Exception:
        # print(f'Инн кпп продавца: {inn_kpp}\nНомер счет фактуры: {invoices}')
        self.data_collection['inn_kpp'] = inn_kpp
        self.data_collection['contract_number'] = contract_number
        self.data_collection['data_table'] = data_table
        try:
            self.data_collection['invoice'] = invoice
            logger.info('JSON data: %s', self.data_collection)
            return self.data_collection
        except IndexError as e:
            logger.error('Ошибка - %s', e)
            return e


class DictToJson:
    """
    Класс для записи хэш-таблицы в json файл
    """
    @staticmethod
    def write_to_json(collection: dict) -> None:
        with open(file='extracted_results/data.json', mode='w', encoding='utf-8') as file:
            json.dump(obj=collection, fp=file, ensure_ascii=False, indent=3)
