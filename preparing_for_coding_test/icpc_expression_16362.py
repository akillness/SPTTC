# (+, -, *, /, and %) 와 space type 의 operator를 ICPC 표현식으로 사용할 때,
# 입력받은 수식이 ICPC 표현식의 오류인지, 적합한지, 부적합한지 판별하는 코드

# eg) a+b*c -> improper / a+(b*c) -> proper / a b*c -> error

# solve) 
# 우선순위 *, /, % 인 경우, () 처리가 되었는지, 잘 되었는지 검사
# ㄴ ( 시작 및 끝 ) 안에 operator 가 있어야함
# +,- 의 경우에도 () 처리 판별식 필요
# Operator 없이 ' '(space) 로 연결된 경우, Error 체크

# qeustion) 과연 아래의 코드가 맞는가..
operators = ['+','-','*','/','%']
def get_operator_cnt(query):

    cnt = 0
    index_sum = 0
    for idx,q in enumerate(query):
        if q in operators:
            cnt += 1
            index_sum += idx

    return cnt, index_sum


def main():
    answer_set = ['proper','improper','error']

    query = input().strip()

    def define_icpc(query):
        
        
        oper_cnt, res = get_operator_cnt(query)
        if oper_cnt > 1 :
            if '(' not in query :
                return answer_set[-2]

        context = ''
        cnt = 0
        pre_brace = 0
        post_brace = 0
        for q in query:
            
            if q == '(': # 연산자 또는 ( 만 가능
                pre_brace+=1

                if cnt < 1:
                    continue

                res = 0
                if context[-1] not in operators and context[-1] != '(':
                    return answer_set[-1]
                elif cnt < 2:
                    return answer_set[-1]
                else:
                    
                    oper_cnt, res = get_operator_cnt(context)
                    res = res%2
                    if oper_cnt > 1:
                        if res:
                            return answer_set[-1]
                    else:
                        if not res:
                            return answer_set[-1]

                # stack.append(context)
                context = ''
                cnt = 0
                
                
            elif q == ')': # 연산자 아닌 값 또는 ) 만 가능
                res = 0
                if context[-1] in operators:
                    return answer_set[-1]
                
                if cnt < 2:
                    return answer_set[-2]
                else:
                    
                    oper_cnt, res = get_operator_cnt(context)
                    res %= 2
                    if context[0] in operators:
                        if oper_cnt < 2 or res:
                            return answer_set[-1]

                    else:
                        if oper_cnt < 2:
                            if not res:
                                return answer_set[-1]
                        else:
                            if res:
                                return answer_set[-1]

                post_brace += 1                     
                
            else:
                if q != ' ':
                    context += q
                    cnt += 1
        
        if post_brace != pre_brace:
            return answer_set[-1]

        return answer_set[0]
            
    print(define_icpc(query))
    
    

if __name__ == "__main__":
    main()