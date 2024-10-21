import re
from logging import getLogger

logger = getLogger(__name__)        # __name__ - имя модуля


class DataExtractionPDF:
    """
    Класс, предназначенный для извлечения, структурирования и записи данных из pdf
    """
    def __init__(self, text: str):
        self._text = text

        self.inn_kpp: str = ''
        self.invoice: str = ''
        self.contract_number: str = ''
        self.amount_without_tax: str = ''
        self.amount_of_tax: str = ''
        self.total_amount: str = ''
        self.data_collection: dict = {}

    @property
    def text(self) -> str:
        return self._text

    def inn_and_kpp_extract(self) -> str:
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
        pattern_invoice_number = re.compile(r'счет-фактура№[^\s]+')
        result = re.search(pattern_invoice_number, self._text.replace(' ', '').lower())
        try:
            self.invoice = result.group(0)
        except AttributeError as e:
            logger.error('Ошибка в атрибуте self.invoice: %s', e)

        logger.info("счет-фактура документа: %s", self.invoice)
        return self.invoice

    def contract_extract(self) -> str:
        # matches = re.findall(r'Договор №(\w+)\s+от\s+([\d.]+)', self._text)
        matches = re.findall(r'Основание|Основания передачи (сдачи) / получения (приемки) \W+\d+|\S+',
                             self._text)
        logger.info('Совпадение контрактов\n%s', matches)

        for match in matches:
            self.contract_number = f'№{match[0]} от {match[1]}'
        return self.contract_number

    def total_sum_extract(self, data_table: list) -> tuple[str] | Exception:
        # Берем последнюю строку таблицы (это строка с суммой)
        try:
            total_sum_list: list = data_table[len(data_table)-1]
            cleaned_totals: list = []
            garbages: tuple = ('', 'Всего к оплате (9)', 'Всего к оплате:', 'Всего к оплате', 'x')
            # очистка от пустых строк (значений)
            for el in total_sum_list:
                if el in garbages:
                    continue
                cleaned_totals.append(el)
            self.total_amount, self.amount_of_tax, self.amount_without_tax = (cleaned_totals[0], cleaned_totals[1],
                                                                              cleaned_totals[2])
            # Добавляем значения в кортеж
            values: tuple = (self.total_amount, self.amount_of_tax, self.amount_without_tax)
            return values
        except IndexError as e:
            logger.error('Ошибка следующего тип (скорее всего, data_table пустой): %s', e)


class DataExtractionImage:
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def seller_extract(self) -> str | Exception:
        pattern_seller = r'Продавец:\s*(\d+\s*"[^"]*"\s*)'
        result = re.search(pattern_seller, self._text)
        try:
            pattern_seller = result.group(0)
            logger.info("Продавец %s", pattern_seller)
            return pattern_seller
        except AttributeError as e:
            logger.error('Ошибка в атрибуте pattern_seller: %s', e)
            return e

    def invoice_extract(self) -> str | Exception:
        pattern_invoice = r'Счет-фактура №\s*(\d+)'
        result = re.search(pattern_invoice, self._text)
        try:
            pattern_seller = result.group(1)
            logger.info("Продавец %s", pattern_seller)
            return pattern_seller
        except AttributeError as e:
            logger.error('Ошибка в атрибуте pattern_seller: %s', e)
            return e

    def inn_and_kpp_extract(self) -> str | Exception:
        pattern_inn_kpp = r'\d{12}|(\d{10}(?:\/\d{9})?)'
        result = re.search(pattern_inn_kpp, self._text.replace(' ', '').lower())
        try:
            inn_kpp = result.group(0)
            logger.info("ИНН/КПП %s", inn_kpp)
            return inn_kpp
        except AttributeError as e:
            logger.error('Ошибка в атрибуте self.inn_kpp: %s', e)
            return e
