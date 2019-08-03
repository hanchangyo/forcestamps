# unique code calculation
import numpy as np


def convertDectoBin(num, n):
    code = np.zeros(n, dtype=np.int)
    # print(code)

    for i in range(n):
        # print(num)
        bit = num % 2
        # print(num % 2)
        # code.append(bit)
        # code = np.append(code, bit)
        code[i] = bit
        num = int(np.floor(num / 2))

    # code.reverse()
    # print(code)
    code = np.flip(code)
    return code


def checkDuplicate(n, code, uniqueCodes):
    isNew = True
    for i in range(n):
        code_shift = np.roll(code, i)
        # print(code_shift)
        for stored in uniqueCodes:
            if stored.tolist() == code_shift.tolist():
                isNew = False
                return isNew
    return isNew


def checkAsymmetry(n, code):
    isAsym = True
    for i in range(1, n):
        code_shift = np.roll(code, i)
        if code.tolist() == code_shift.tolist():
            isAsym = False
            return isAsym
    return isAsym


def checkRepeatingTwoZeros(n, code):
    isNotRepeat = True
    code = np.append(code, code[0])
    for i in range(n):
        if code[i] == 0 and code[i + 1] == 0:
            isNotRepeat = False
            return isNotRepeat
    return isNotRepeat


n = 13

# num = 15
# print(convertDectoBin(num, n))

uniqueCodes = []
uniqueCodes_three_ones = []
uniqueCodes_asym = []
uniqueCodes_no_two_zeros_in_a_row = []

codeList = [[0]]
for msg in range(2 ** n):
    print('%d / %d' % (msg, 2 ** n))
    code = convertDectoBin(msg, n)
    # isNew = True

    isNew = checkDuplicate(n, code, uniqueCodes)

    if isNew:
        uniqueCodes.append(code)
        if np.sum(code) > 2:
            uniqueCodes_three_ones.append(code)
            isAsym = checkAsymmetry(n, code)
            if isAsym:
                uniqueCodes_asym.append(code)
                isNotRepeat = checkRepeatingTwoZeros(n, code)
                if isNotRepeat:
                    uniqueCodes_no_two_zeros_in_a_row.append(code)


# print(uniqueCodes)
# for code in uniqueCodes:
#     print(code)


for code in uniqueCodes_no_two_zeros_in_a_row:
    print(code)

print(len(uniqueCodes))

print(len(uniqueCodes_three_ones))

print(len(uniqueCodes_asym))

print(len(uniqueCodes_no_two_zeros_in_a_row))

'''
# uniqueCodes = [np.zeros(15, dtype=np.int)]
# codeList = [[0]]

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
'''