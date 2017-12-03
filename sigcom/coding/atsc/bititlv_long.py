'''
Source of the bil patterns: http://www.atsc.org/wp-content/uploads/2016/10/A322-2016-Physical-Layer-Protocol.pdf
'''

import numpy as np

def bititlv_long(M, code_rate_id):
    if M==4:
        if code_rate_id == [2, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [3, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [4, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [5, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [6, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [7, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [8, 15]:
            bil = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,166,168,170,172,174,176,178,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,167,169,171,173,175,177,179]
        elif code_rate_id == [9, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [10, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [11, 15]:
            assert False, 'Implement!'
        elif code_rate_id == [12, 15]:
            assert False, 'Implement!'        
        elif code_rate_id == [13, 15]:
            assert False, 'Implement!'
    else:
        assert False, 'Implement!'
    return np.array(bil, dtype=np.int32)