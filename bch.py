# coding: utf-8
# bch code test

import numpy as np


def bchEncode15_7(msg):
    # bch generator polynomial for n = 15, k = 7, t = 2
    # x^8 + x^7 + x^6 + x^4 + 1
    n = 15
    k = 7
    genPoly = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1])

    # encode message
    # bit shift message bits by code length
    msgBits = bitfield(msg, k)
    msgBitsShifted = np.concatenate((np.zeros(n - k, dtype=np.int), msgBits))
    # print(msgBitsShifted)

    # divide by generator polynomial
    Q = np.zeros(k, dtype=np.int)
    R = np.zeros(n, dtype=np.int)
    R[::] = msgBitsShifted[::][::-1]
    # print(R)
    # print(genPoly[-1])

    for i in range(k):
        # print(i)
        if genPoly[::-1][0] == 1 and R[i] == 1:
            Q[i] = 1
            R = (
                R - np.roll(np.concatenate((genPoly[::-1], np.zeros(k - 1, dtype=np.int))), i)
            ) % 2
            # print(Q)
            # print(R)

    # remainder is parity bits
    R = R[k:][::-1]
    code = np.concatenate((R, msgBits))

    return code


def bitfield(n, length):
    if n > 2 ** length - 1:
        print('message is out of range!')
        return False
    else:
        bitArray = np.zeros(length, dtype=np.int)
        # [2:] to chop off the "0b" part
        temp = [int(digit) for digit in bin(n)[2:]]
        bitArray[:len(temp)] = temp[::-1]
        return bitArray


def GF16_p_v(power):
    # power to vector representation of GF16
    if power == 0:
        vector = np.array([0, 0, 0, 1], dtype=np.int)
    elif power == 1:
        vector = np.array([0, 0, 1, 0], dtype=np.int)
    elif power == 2:
        vector = np.array([0, 1, 0, 0], dtype=np.int)
    elif power == 3:
        vector = np.array([1, 0, 0, 0], dtype=np.int)
    elif power == 4:
        vector = np.array([0, 0, 1, 1], dtype=np.int)
    elif power == 5:
        vector = np.array([0, 1, 1, 0], dtype=np.int)
    elif power == 6:
        vector = np.array([1, 1, 0, 0], dtype=np.int)
    elif power == 7:
        vector = np.array([1, 0, 1, 1], dtype=np.int)
    elif power == 8:
        vector = np.array([0, 1, 0, 1], dtype=np.int)
    elif power == 9:
        vector = np.array([1, 0, 1, 0], dtype=np.int)
    elif power == 10:
        vector = np.array([0, 1, 1, 1], dtype=np.int)
    elif power == 11:
        vector = np.array([1, 1, 1, 0], dtype=np.int)
    elif power == 12:
        vector = np.array([1, 1, 1, 1], dtype=np.int)
    elif power == 13:
        vector = np.array([1, 1, 0, 1], dtype=np.int)
    elif power == 14:
        vector = np.array([1, 0, 0, 1], dtype=np.int)
    elif power == False:
        vector = np.array([0, 0, 0, 0], dtype=np.int)
    else:
        vector = 0

    return vector


def GF16_v_p(vector):
    # vector to power representation of GF16
    if vector.tolist() == [0, 0, 0, 1]:
        power = 0
    elif vector.tolist() == [0, 0, 1, 0]:
        power = 1
    elif vector.tolist() == [0, 1, 0, 0]:
        power = 2
    elif vector.tolist() == [1, 0, 0, 0]:
        power = 3
    elif vector.tolist() == [0, 0, 1, 1]:
        power = 4
    elif vector.tolist() == [0, 1, 1, 0]:
        power = 5
    elif vector.tolist() == [1, 1, 0, 0]:
        power = 6
    elif vector.tolist() == [1, 0, 1, 1]:
        power = 7
    elif vector.tolist() == [0, 1, 0, 1]:
        power = 8
    elif vector.tolist() == [1, 0, 1, 0]:
        power = 9
    elif vector.tolist() == [0, 1, 1, 1]:
        power = 10
    elif vector.tolist() == [1, 1, 1, 0]:
        power = 11
    elif vector.tolist() == [1, 1, 1, 1]:
        power = 12
    elif vector.tolist() == [1, 1, 0, 1]:
        power = 13
    elif vector.tolist() == [1, 0, 0, 1]:
        power = 14
    else:
        power = -np.Inf

    return power


def calculateErrorLocation(S_1_vec, S_3_vec):
    # = 1 + S_1 * z + (S_1^3 + S_3)/S_1 * z^2
    S_1__3 = (GF16_v_p(S_1_vec) * 3) % 15
    # print('S_1^3', S_1__3)
    S_3 = GF16_v_p(S_3_vec)
    # print('S_3', S_3)

    if S_1_vec.tolist() == [0, 0, 0, 0] and S_3_vec.tolist() != [0, 0, 0, 0]:
        z_2 = (S_3 - GF16_v_p(S_1_vec)) % 15
    elif S_1_vec.tolist() != [0, 0, 0, 0] and S_3_vec.tolist() == [0, 0, 0, 0]:
        z_2 = (S_1__3 - GF16_v_p(S_1_vec)) % 15        
    else:
        z_2 = (GF16_v_p((GF16_p_v(S_1__3) + S_3_vec) % 2) - GF16_v_p(S_1_vec)) % 15

    z_1 = GF16_v_p(S_1_vec)

    return z_1, z_2


def calculateSyndrome(rec):
    # calculate S_1 and S_3
    S_1_vec = np.zeros(4, dtype=np.int)
    for i in range(len(rec)):
        if rec[i] == 1:
            S_1_vec += GF16_p_v(i)
    S_1_vec = S_1_vec % 2

    S_3_vec = np.zeros(4, dtype=np.int)
    for i in range(len(rec)):
        if rec[i] == 1:
            S_3_vec += GF16_p_v((i * 3) % 15)
    S_3_vec = S_3_vec % 2

    # print('Syndrome1', S_1_vec)
    # print('Syndrome3', S_3_vec)

    # if all syndromes are zero
    if S_1_vec.tolist() == [0, 0, 0, 0] and S_3_vec.tolist() == [0, 0, 0, 0]:
        error = False
    else:
        error = True

    if error:
        # sigma(z) = 1 + z_1 * z + z_2 * z^2
        # = 1 + S_1 * z + (S_1^3 + S_3)/S_1 * z^2
        z_1, z_2 = calculateErrorLocation(S_1_vec, S_3_vec)
        # print('z1', z_1)
        # print('z2', z_2)

        errorPos = []
        for i in range(15):
            a = (np.array([0, 0, 0, 1], dtype=np.int) +
                 GF16_p_v((z_1 + i) % 15) + GF16_p_v((z_2 + i * 2) % 15)) % 2
            if a.tolist() == [0, 0, 0, 0]:
                errorPos.append((15 - i) % 15)
        # print(errorPos)

        # print(error)
        for i in errorPos:
            rec[i] = (rec[i] + 1) % 2

    return rec


def bchDecode15_7(code):
    
    n = 15
    k = 7
    # FEC & decoded received code
    codeCorrected = calculateSyndrome(code)

    msgBin = codeCorrected[8:]
    msgDec = np.sum(msgBin * np.array([1, 2, 4, 8, 16, 32, 64]))

    return msgDec




    # FEC & decode received code. Recognize ID.
    # uniqueCodes = [
    #     np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.int), 
    #     np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.int), 
    #     np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int), 
    #     np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int), 
    #     np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int)
    # ]

'''
t = 2
# msg = 126

# uniqueCodes = []
uniqueCodes = [np.zeros(15, dtype=np.int)]
codeList = [[0]]
for msg in range(128):
    code = bchEncode15_7(msg)
    isNew = True

    for i in range(15):
        code_shift = np.roll(code, i)
        # print(code_shift)
        for stored in uniqueCodes:
            if stored.tolist() == code_shift.tolist():
                isNew = False
                # stored.append(code_shift)
        for c in codeList:
            if c[0] == bchDecode15_7(code_shift):
                c.append(bchDecode15_7(code))
                break

    if isNew:
        uniqueCodes.append(code)
        print(bchDecode15_7(code))
        codeList.append([])
        codeList[-1].append(bchDecode15_7(code))

for i in range(len(codeList)):
    codeList[i] = list(set(codeList[i]))

print(uniqueCodes)
print(codeList)

for i in uniqueCodes:
    codeString = ''
    for j in i:
        if j == 1:
            codeString += '1'
        else:
            codeString += '0'
    print(codeString)

# for i in uniqueCodes:
    # print(np.sum(i))
print(len(uniqueCodes))



# print(bitfield(msg, 7))
# code = bchEncode15_7(msg)
# print(code)
# for error_i in range(15):
#     for error_j in range(15):
#         if error_i != error_j:
#             errorVector = np.array(
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int)
#             errorVector[error_i] = 1
#             errorVector[error_j] = 1
#             codeError = (code + errorVector) % 2
#             codeCorrected = calculateSyndrome(codeError)
#             print(codeCorrected)
'''