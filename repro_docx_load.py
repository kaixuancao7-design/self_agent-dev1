import traceback
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

try:
    docs = UnstructuredWordDocumentLoader('data/AI Agent热点文章周报_26_4_8-_4_14.docx').load()
    print('loaded', len(docs))
except Exception:
    traceback.print_exc()
