from pdf2image import convert_from_path


class Converter:
    def __init__(self,):
        """
        Скорее всего, также нужно будет указать путь до исполнительного файла, как и в классе Img
        def __init__(self, path_dir: str, exe_file: str):
            self.path_dir = path_dir
            self.exe_file = exe_file
        """
        pass

    def convert_from_pdf_to_img(self, pdf_path: str, to_save: str) -> None:
        # Второй параметр влияет на качество изображения (при необходимости можно поставить выше значение)
        pages = convert_from_path(pdf_path=pdf_path, dpi=200)
        # Сохраняем каждое изображение в PNG формате
        for i, page in enumerate(pages):
            page.save(f'{to_save}_{i + 1}.png', 'PNG')