from sigcom.tx.atsc_NUCs_LG import atsc_NUCs_LG
from sigcom.tx.atsc_NUCs_ETRI import atsc_NUCs_ETRI
from sigcom.tx.atsc_NUCs_SS import atsc_NUCs_SS
from sigcom.tx.atsc_NUCs_NERC import atsc_NUCs_NERC
from sigcom.tx.atsc_NUCs_PANA import atsc_NUCs_PANA

def alphabet(M, CR):
    if M == 4:
        raise Exception('Implement!')
    elif M==16:
        d = {
            2:atsc_NUCs_ETRI,
            3:atsc_NUCs_ETRI,
            4:atsc_NUCs_ETRI,
            5:atsc_NUCs_LG,
            6:atsc_NUCs_SS,
            7:atsc_NUCs_NERC,
            8:atsc_NUCs_LG,
            9:atsc_NUCs_LG,
            10:atsc_NUCs_SS,
            11:atsc_NUCs_SS,
            12:atsc_NUCs_SS,
            13:atsc_NUCs_SS
        }
    elif M==64:
        d = {
            2:atsc_NUCs_ETRI,
            3:atsc_NUCs_ETRI,
            4:atsc_NUCs_ETRI,
            5:atsc_NUCs_SS,
            6:atsc_NUCs_SS,
            7:atsc_NUCs_LG,
            8:atsc_NUCs_SS,
            9:atsc_NUCs_LG,
            10:atsc_NUCs_LG,
            11:atsc_NUCs_SS,
            12:atsc_NUCs_SS,
            13:atsc_NUCs_LG
        }
    elif M==256:
        d = {
            2:atsc_NUCs_ETRI,
            3:atsc_NUCs_ETRI,
            4:atsc_NUCs_ETRI,
            5:atsc_NUCs_LG,
            6:atsc_NUCs_LG,
            7:atsc_NUCs_LG,
            8:atsc_NUCs_LG,
            9:atsc_NUCs_SS,
            10:atsc_NUCs_LG,
            11:atsc_NUCs_LG,
            12:atsc_NUCs_SS,
            13:atsc_NUCs_LG
        }
    elif M==1024:
        d = {
            2:atsc_NUCs_ETRI,
            3:atsc_NUCs_ETRI,
            4:atsc_NUCs_ETRI,
            5:atsc_NUCs_LG,
            6:atsc_NUCs_SS,
            7:atsc_NUCs_LG,
            8:atsc_NUCs_SS,
            9:atsc_NUCs_LG,
            10:atsc_NUCs_LG,
            11:atsc_NUCs_LG,
            12:atsc_NUCs_SS,
            13:atsc_NUCs_SS
        }
    elif M==4096:
        d = {
            2:atsc_NUCs_ETRI,
            3:atsc_NUCs_ETRI,
            4:atsc_NUCs_ETRI,
            5:atsc_NUCs_ETRI,
            6:atsc_NUCs_NHK,
            7:atsc_NUCs_PANA,
            8:atsc_NUCs_NHK,
            9:atsc_NUCs_NHK,
            10:atsc_NUCs_NHK,
            11:atsc_NUCs_NHK,
            12:atsc_NUCs_NHK,
            13:atsc_NUCs_NHK
        }

    return d[CR[0]](M, CR)
