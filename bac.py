import numpy as np

class BACEncoder:
    def __init__(self):
        self.L = 0
        self.R = 1 << 32
        self.cod = []
        self.buf = 0
        self.cnt = 0

    def encode(self, bit, zero_prob):
        base = 1 << 16
        ZR = (self.R * int(base * zero_prob)) >> 16
        OR = self.R - ZR
        if bit == 0:
            self.R = ZR
        else:
            self.R = OR
            self.L += ZR

        # Renorm
        if self.L >= 1 << 32:
            self.buf += 1
            self.L &= 0xffffffff
            if self.cnt > 0:
                self.cod.append(self.buf)
                for _ in range(self.cnt - 1):
                    self.cod.append(0)
                self.buf = 0
                self.cnt = 0
        while (self.R < (1 << 24)):
            if self.L < (0xff << 24):
                self.cod.append(self.buf)
                for _ in range(self.cnt):
                    self.cod.append(0xff)
                self.cnt = 0
                self.buf = (self.L >> 24) & 0xff
            else:
                self.cnt += 1

            self.R <<= 8
            self.L = (self.L << 8) & 0xffffffff

    def flush(self):
        c = 0xff
        if self.L >= 1 << 32:
            self.buf += 1
            c = 0
        self.cod.append(self.buf)
        for _ in range(self.cnt):
            self.cod.append(c)
        for _ in range(4):
            self.cod.append((self.L >> 24) & 0xff)
            self.L <<= 8

class BACDecoder:
    def __init__(self, cod):
        C = 0
        self.R = 1 << 32
        code = 0
        read_pos = 1
        for _ in range(4):
            code = (code << 8) | cod[read_pos]
            read_pos += 1
        self.read_pos = read_pos
        self.cod = cod
        self.C = code
        self.L = 0

    def decode(self, zero_prob):
        base = 1 << 16
        ZR = (self.R * int(base * zero_prob)) >> 16
        OR = self.R - ZR
        bit = 0
        if self.L + ZR <= self.C:
            bit = 1
            self.R = OR
            self.C -= ZR
        else:
            bit = 0
            self.R = ZR

        # Renorm
        while self.R < (1 << 24):
            self.R <<= 8
            self.C = ((self.C << 8) + self.cod[self.read_pos]) & 0xffffffff
            self.read_pos += 1

        return bit

def test_stat():
    source = []

    # zero_prb
    prob = 1.0 / 4
    n = 4

    for _ in range(10000):
        b = np.random.choice(n)
        if b == 0:
            source.append(0)
        else:
            source.append(1)

    encoder = BACEncoder()
    for bit in source:
        encoder.encode(bit, prob)
    encoder.flush()
    print(len(encoder.cod))

    decoder = BACDecoder(encoder.cod)
    decod = []
    for i in range(10000):
        decod.append(decoder.decode(prob))

    print(source == decod)

test_stat()
