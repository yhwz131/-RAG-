#!/usr/bin/env python3
"""Extract text from .doc file using olefile + raw parsing"""
import olefile
import re
import sys

def extract_doc_text(path):
    ole = olefile.OleFileIO(path)
    
    # Read the WordDocument stream
    word_stream = ole.openstream('WordDocument').read()
    
    # Read the data from 1Table or 0Table
    table_name = '1Table' if ole.exists('1Table') else '0Table'
    table_stream = ole.openstream(table_name).read()
    
    # The text in .doc is stored as UTF-16LE in the WordDocument stream
    # Try to find readable Chinese text blocks
    # Method: decode the whole stream as utf-16-le and filter
    
    # Try reading from the main document body
    # The FIB (File Information Block) starts at offset 0
    # ccpText is at offset 0x4C (76) in the FIB
    import struct
    
    # Get the beginning of text offset from FIB
    # fcMin is at offset 0x18 (24) - 4 bytes
    fc_min = struct.unpack_from('<I', word_stream, 0x18)[0]
    # ccpText is at offset 0x4C (76) - 4 bytes  
    ccp_text = struct.unpack_from('<i', word_stream, 0x4C)[0]
    
    print(f"DEBUG: fc_min={fc_min}, ccp_text={ccp_text}", file=sys.stderr)
    
    if ccp_text > 0:
        # Text is in the WordDocument stream starting at fc_min
        text_bytes = word_stream[fc_min:fc_min + ccp_text * 2]
        text = text_bytes.decode('utf-16-le', errors='ignore')
    else:
        # Fallback: scan entire stream for text
        text = word_stream.decode('utf-16-le', errors='ignore')
    
    # Clean up: remove control characters but keep Chinese, punctuation, etc.
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    ole.close()
    return text.strip()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'docs1/2251115-江政宾（开题报告）.doc'
    text = extract_doc_text(path)
    print(text)
