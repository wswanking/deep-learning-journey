"""Fixed UTF-8 byte vocabulary 0..255, no learned vocabulary or reserved special tokens."""
VOCAB_SIZE=256

def encode(text: str) -> list[int]:
    return list(text.encode('utf-8'))

def decode(ids, *, errors='replace') -> str:
    """Incomplete generated UTF-8 sequences are replaced by default, explicitly reported in evaluation."""
    return bytes(int(value) for value in ids).decode('utf-8',errors=errors)
