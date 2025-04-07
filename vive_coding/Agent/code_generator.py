from typing import List, Callable, Any, Dict, Tuple
from functools import wraps
import re
import os
import logging
from datetime import datetime
from openai import OpenAI
from file_manager import FileManager

# tool 데코레이터는 ToolCallingAgent에서 직접 사용하지 않으므로 주석 처리 또는 삭제 가능
# def tool(func: Callable) -> Callable:
#     """데코레이터: 함수를 도구로 등록"""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs)
#     wrapper.is_tool = True
#     return wrapper

class CodeGeneratorAgent:
    def __init__(self, client: OpenAI):
        """초기화 함수
        
        Args:
            client (OpenAI): OpenAI API 클라이언트
        """
        self.client = client
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 언어 감지 키워드 (소문자)
        self.language_keywords = {
            "python": ["python", "파이썬"],
            "javascript": ["javascript", "js", "자바스크립트"],
            "java": ["java", "자바"],
            "c++": ["c++", "cpp", "씨쁠쁠"],
            "c": ["c", "씨", "씨언어"],
            "c#": ["c#", "cs", "csharp", "씨샵"],
            "go": ["go", "golang", "고"],
            "rust": ["rust", "러스트"],
            "ruby": ["ruby", "루비"],
            "php": ["php"],
            "typescript": ["typescript", "ts", "타입스크립트"],
            "swift": ["swift", "스위프트"],
            "kotlin": ["kotlin", "코틀린"],
            "r": ["r", "알"]
            # 필요에 따라 언어 추가
        }
        # 코드 생성 감지 키워드
        self.codegen_keywords = ["코드", "프로그램", "작성", "만들", "짜줘", "generate", "create", "write", "code", "program"]
        # 코드 실행 감지 키워드
        self.execution_keywords = ["실행", "돌려", "run", "execute"]

        # 언어 이름 -> 확장자 매핑 (FileManager 활용)
        # FileManager.LANGUAGE_MAP의 key(확장자)와 value(언어이름)를 뒤집음
        self.lang_to_ext = {v: k for k, v in FileManager.LANGUAGE_MAP.items()}
        # FileManager에 없는 언어 수동 추가 (예시)
        self.lang_to_ext.setdefault('javascript', '.js') 
        self.lang_to_ext.setdefault('typescript', '.ts')
        self.lang_to_ext.setdefault('swift', '.swift')
        self.lang_to_ext.setdefault('kotlin', '.kt')
        # C# 확장자 추가 (.cs는 FileManager에 이미 있을 수 있지만 확인)
        self.lang_to_ext.setdefault('c#', '.cs')

        # pip 설치가 필요할 수 있는 일반적인 패키지 목록 (표준 라이브러리 제외)
        self.common_pip_packages = {
            'requests', 'numpy', 'pandas', 'matplotlib', 'scipy', 'pygame', 
            'beautifulsoup4', 'bs4', 'selenium', 'pillow', 'PIL', 'flask', 
            'django', 'sqlalchemy', 'fastapi', 'tensorflow', 'keras', 
            'torch', 'scikit-learn', 'sklearn' 
            # tkinter는 시스템 설치 필요하므로 제외
        }

    def _detect_language_and_request(self, task: str) -> Tuple[str | None, bool, bool]:
        """작업 문자열에서 언어, 코드 생성, 실행 요청 여부 감지"""
        task_lower = task.lower()
        detected_language = None
        is_codegen_request = any(keyword in task_lower for keyword in self.codegen_keywords)
        is_execution_request = any(keyword in task_lower for keyword in self.execution_keywords)
        
        # 코드 생성 요청이 없으면 언어 감지 불필요
        if not is_codegen_request:
            return None, False, False

        for lang_name, keywords in self.language_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                detected_language = lang_name
                break 
                
        # 언어가 특정되지 않았지만 코드 생성 요청은 있을 수 있음 (기본 Python으로 처리할 수도 있음)
        # 여기서는 언어가 명시된 경우만 코드 생성 처리
        if detected_language is None:
             is_codegen_request = False # 언어 없으면 코드 생성 불가 처리

        return detected_language, is_codegen_request, is_execution_request
        
    def _clean_llm_code_output(self, code_content: str, language: str) -> str:
        """LLM 응답에서 코드 블록 마크다운 제거"""
        # 정규표현식으로 ```language ... ``` 또는 ``` ... ``` 제거 시도
        pattern = rf"^```(?:{language}|\w*)?\s*\n(.*?)\n```$"
        match = re.match(pattern, code_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # 간단히 앞뒤 ``` 제거 시도 (마크다운이 불완전할 경우 대비)
        if code_content.startswith("```") and code_content.endswith("```"):
             lines = code_content.splitlines()
             if len(lines) > 1:
                 # 첫 줄(```language)과 마지막 줄(```) 제거
                 return "\n".join(lines[1:-1]).strip()
        return code_content # 제거할 패턴 없으면 원본 반환

    def _find_python_imports(self, code: str) -> List[str]:
        """Python 코드에서 import 문을 찾아 모듈 이름 리스트 반환 (간단한 정규식 사용)"""
        # import module / from module import ... / import module as alias
        import_pattern = r'^\s*(?:import|from)\s+([\w\.]+)'
        matches = re.findall(import_pattern, code, re.MULTILINE)
        # 점(.)으로 시작하는 상대 임포트 등은 제외하고, 첫 부분만 추출
        modules = {m.split('.')[0] for m in matches if m and not m.startswith('.')}
        return list(modules)

    def _check_required_packages(self, modules: List[str]) -> List[str]:
        """임포트된 모듈 중 설치가 필요할 수 있는 패키지 목록 반환"""
        required = []
        for module in modules:
            # bs4는 beautifulsoup4로 설치해야 함
            package_name = 'beautifulsoup4' if module == 'bs4' else module
            # scikit-learn은 sklearn으로 설치해야 함
            package_name = 'scikit-learn' if module == 'sklearn' else package_name
            # PIL은 Pillow로 설치해야 함
            package_name = 'Pillow' if module == 'PIL' else package_name
            
            if package_name in self.common_pip_packages:
                required.append(package_name)
        return sorted(list(set(required))) # 중복 제거 및 정렬

    def run(self, task: str, search_context: str | None = None, print_results: bool = False) -> Dict[str, Any]:
        """작업 실행 (다국어 코드 생성, 패키지 감지, 실행 요청 감지)
        
        Args:
            task (str): 사용자의 원본 작업 요청
            search_context (str | None, optional): 웹 검색 결과 (요약). Defaults to None.
            print_results (bool, optional): 결과를 콘솔에 출력할지 여부. Defaults to False.
            
        Returns:
            Dict[str, Any]: 작업 처리 결과 딕셔너리
        """
        
        detected_language, is_codegen_request, is_execution_request = self._detect_language_and_request(task)

        if is_codegen_request and detected_language:
            logging.info(f"LLM 코드 생성 요청 ({detected_language}): {task}")
            
            # 시스템 프롬프트 구성
            system_prompt = f"You are a code generation assistant. Generate *only* the raw code in the requested language ({detected_language}) based on the user's request. Do not include any explanations, comments, markdown formatting (like ```{detected_language}), or introductory phrases. Just output the code itself."
            if search_context:
                system_prompt += f"\n\nUse the following search results as context if relevant:\n--- SEARCH CONTEXT ---\n{search_context}\n--- END SEARCH CONTEXT ---"
                logging.info("코드 생성 시 검색 컨텍스트 사용")

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate {detected_language} code for the following request: {task}"}
                    ],
                    temperature=0.5, 
                )
                raw_code_content = response.choices[0].message.content.strip()
                code_content = self._clean_llm_code_output(raw_code_content, detected_language)
                
                required_packages = []
                if detected_language == 'python':
                    imported_modules = self._find_python_imports(code_content)
                    required_packages = self._check_required_packages(imported_modules)
                    if required_packages:
                         logging.info(f"감지된 필요 패키지: {required_packages}")

                if not code_content:
                    logging.warning(f"LLM이 빈 {detected_language} 코드를 반환했거나 파싱 실패.")
                    code_content = f"# LLM이 {detected_language} 코드를 생성하지 못했거나 응답 파싱에 실패했습니다." 
                
                file_extension = self.lang_to_ext.get(detected_language, '.txt')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_code_{timestamp}{file_extension}"
                file_path = os.path.join(self.output_dir, filename)
                
                saved_message = ""
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(code_content)
                    saved_message = f"코드가 다음 경로에 저장되었습니다: {file_path}"
                except Exception as e:
                    saved_message = f"코드 저장 중 오류 발생: {e}"
                    logging.error(saved_message)
                    file_path = None # 저장 실패 시 경로 없음
                    
                result = {
                    "task": task,
                    "result_type": "code_generation", 
                    "language": detected_language,
                    "generated_code": code_content,
                    "saved_path_message": saved_message,
                    "saved_file_path": file_path, # 실제 파일 경로 추가
                    "execute_request": is_execution_request, # 실행 요청 여부 추가
                    "required_packages": required_packages, # 필요 패키지 정보 추가
                    "status": "success"
                }
                return result
                
            except Exception as e:
                logging.error(f"OpenAI API 호출 중 오류 발생 ({detected_language}): {e}")
                return {
                    "task": task,
                    "result_type": "error",
                    "result": f"LLM {detected_language} 코드 생성 중 오류 발생: {e}",
                    "status": "failed"
                }

        # --- 코드 생성이 아니거나 언어를 특정할 수 없는 경우 ---
        result = {
            "task": task,
            "result_type": "unknown",
            "result": "요청을 처리할 수 없습니다. 명확한 명령어를 사용하거나 지원되는 언어(Python, C++, Java 등)와 함께 코드 생성을 요청해주세요.",
            "status": "failed"
        }
            
        return result 