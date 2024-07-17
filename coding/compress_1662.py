
# K(Q)로 입력된 문자열을 압축할 수 있다
# K는 한자리 숫자, Q는 0자리수의 이상 문자열
# 입력 문자열은 (, ), 0-9사이의 숫자
# 출력은 문자열의 길이(Length)

def main():
    seq = input().strip()

    def solve(sequence):
        stack = []
        text = ''
        for i,c in enumerate(sequence):
            if c == '(':
                # ( 기준 Q(text) 와 K(개수) 값 Stack
                stack.append([sequence[i-1],text[:-1]])
                text = ''
                # cnt = 0
            elif c == ')':
                K,Q = stack.pop()
                text = Q + ''.join(int(K)*[text])
                
            else:
                text += c
                # cnt +=1

        print(len(text))
        return text
    
    print(solve(seq))

    # texts = ' '.join(input().split())

    # def get_sequence_length(sequence):
    #     if sequence.find('(') > 0:
    #         recon_seq = ''
    #         inputs = sequence.replace(')','').split('(')
    #         chr = inputs[-1]
    #         for idx in range(len(inputs)-2,-1,-1):
    #             part_seq = [chr] * int(inputs[idx][-1])
    #             part_seq = ''.join(part_seq)                
    #             recon_seq = inputs[idx][:-1] + part_seq
    #             chr = recon_seq
    #         print(recon_seq)
    #         return len(recon_seq)
    #     else:
    #         print(sequence)
    #         return len(sequence)


    # res = get_sequence_length(texts)
    # print(res)


if __name__ =="__main__":
    main()