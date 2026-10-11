from .MCN import MCN
from .UMC_ICLA import UMCICLA
from .BERT_TEXT import BERT_TEXT
multimodal_methods_map = {
    'mcn': MCN,
    'umc-icla': UMCICLA,
    'text': BERT_TEXT,
}
