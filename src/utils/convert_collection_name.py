import re
from unidecode import unidecode

def convert_collection_name(raw_name: str) -> str:
    name = unidecode(raw_name)

    name = name.lower()
    name = re.sub(r'[^\w\s-]', '', name)  # Loại bỏ ký tự không phải chữ/số/gạch
    name = re.sub(r'[\s\-]+', '_', name)  # Thay khoảng trắng/gạch nối thành "_"

    name = name.strip('_')
    
    if not (3 <= len(name) <= 512):
        raise ValueError("Tên sau khi chuyển đổi không hợp lệ (độ dài 3-512).")

    return name
