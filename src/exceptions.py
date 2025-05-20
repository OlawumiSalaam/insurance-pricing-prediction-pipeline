import sys

def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message with information about 
    the file name, line number, and error itself.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name}, line: {line_number}, error: {str(error)}"
    return error_message


class CustomException(Exception):
    """
    Custom exception class that uses the error_message_detail function to provide detailed error info.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
