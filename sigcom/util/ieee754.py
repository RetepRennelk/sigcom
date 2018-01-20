import numpy as np


class IEEE754():
    def __init__(self, f):
        self.f = f
        self.uint32bits = np.float32(f).view(np.uint32)
        self.sign_bit = self.uint32bits >> 31
        exponent = (self.uint32bits << 1) & 0xFFFFFFFF
        self.exponent = exponent >> 24
        self.mantissa = self.uint32bits & 0x7FFFFF

    def print(self):
        print('sign:', self.sign_bit)
        print('expo:', self.exponent)
        print('mant:', self.mantissa)
        str = '{:032b}'.format(self.uint32bits)
        out_str = str[0] + ' '
        out_str += str[1:9] + ' '
        out_str += str[9:]
        print(out_str)


if __name__ == '__main__':
    init()
    x = IEEE754(1.0)
    x.print()

