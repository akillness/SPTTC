

# filter 와 join 이용한 문자열 제거
def solution(str_list, ex):
    return ''.join(filter(lambda x: ex not in x,str_list))


# set 함수를 이용한, 중복판별
a = 5
b = 6
c = 1

print(list(set([a,b,c])))

# set 함수를 이용한 교집합,합집합
a = [1,2,3]
b = [3,4,5]
aib = list(set(a)&set(b))
asb = list(set(a)|set(b))

print(aib)
print(asb)