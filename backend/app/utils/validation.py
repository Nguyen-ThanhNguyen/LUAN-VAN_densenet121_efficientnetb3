ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def safe_extension(filename: str) -> str:
    ext = filename.rsplit(".", 1)[1].lower()
    return "jpg" if ext == "jpeg" else ext


def validate_upload(file_storage):
    if file_storage is None:
        return False, "Không tìm thấy file upload."
    if not file_storage.filename:
        return False, "Tên file rỗng."
    if not allowed_file(file_storage.filename):
        return False, "Định dạng không hợp lệ. Chỉ hỗ trợ PNG, JPG, JPEG."
    return True, ""
