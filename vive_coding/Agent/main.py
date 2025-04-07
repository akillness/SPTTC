from typing import List, Dict, Any
import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import json
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from file_manager import FileManager
from code_executor import CodeExecutor
import subprocess
from code_generator import CodeGeneratorAgent
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent.log', encoding='utf-8')
    ]
)

def explore_directory(dir_path: str = '.') -> Dict:
    """디렉토리 내용을 탐색합니다."""
    return FileManager.explore_directory(dir_path)

def manage_files(action: str, path: str, new_path: str = None) -> str:
    """파일 관리 작업을 수행합니다."""
    return FileManager.manage_files(action, path, new_path)

def execute_code(code: str, language: str = "python") -> str:
    """코드 블록을 실행합니다."""
    return CodeExecutor.execute_code(code, language)

class AgentAI:
    def __init__(self, name: str, description: str, memory_limit: int = 10):
        """초기화 함수
        
        Args:
            name (str): 에이전트의 이름
            description (str): 에이전트의 설명
            memory_limit (int, optional): 메모리에 저장할 최대 대화 수. Defaults to 10.
        """
        load_dotenv()  # .env 파일에서 환경 변수 로드
        
        self.name = name
        self.description = description
        self.memory: List[Dict[str, Any]] = []
        self.memory_limit = memory_limit
        
        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI()
        logging.info(f"{self.name} 에이전트가 성공적으로 초기화되었습니다.")
        
        # 도구 등록 및 에이전트 초기화 (tools, system_prompt 제거)
        self.agent = CodeGeneratorAgent(
            client=self.client
        )
        
        # 시스템 메시지 설정 (이 부분은 AgentAI의 설명 용도로 유지 가능)
        self.system_message = f"""당신은 {name}이라는 이름의 AI 에이전트입니다.
{description}

당신은 다음과 같은 작업들을 수행할 수 있습니다:
1. 파일 관리
2. 코드 실행
3. 검색
4. 계산
각 작업을 수행할 때는 적절한 도구를 선택하여 사용하겠습니다."""

    def run_interactive(self):
        """대화형 모드로 실행"""
        print(f"\n=== {self.name} 시작 ===")
        print(f"{self.description}")

        while True:
            try:
                # 사용자 입력 받기
                user_input = input("\n명령어를 입력하세요 (도움말: help): ").strip()
                
                # 종료 명령 처리
                if user_input.lower() in ['exit', '종료', 'quit', 'q']:
                    print("프로그램을 종료합니다.")
                    break                
                
                # 작업 실행
                if user_input:
                    print("\n=== 작업 실행 ===")
                    result_message = self.run_task(user_input)
                    print(f"\n결과:\n{result_message}")
                    print("=" * 50)
            
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                break

    def _manage_memory(self):
        """메모리 관리 함수"""
        if len(self.memory) > self.memory_limit:
            removed = self.memory[:-self.memory_limit]
            self.memory = self.memory[-self.memory_limit:]
            logging.info(f"{len(removed)}개의 오래된 대화가 메모리에서 제거되었습니다.")
    
    def _search(self, query: str) -> str:
        """인터넷 검색을 수행하고 결과를 요약하여 반환"""
        logging.info(f"검색 및 요약 시도: {query}")
        
        search_results_text = []
        urls_processed = []
        logging.info("구글 검색 시작...")
        
        try:
            # 검색 결과 가져오기 (최대 3개 URL)
            for url in search(query, stop=3):
                urls_processed.append(url) # 어떤 URL을 처리했는지 기록
                try:
                    logging.info(f"URL 처리 중: {url}")
                    response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) # User-Agent 추가
                    response.raise_for_status() # HTTP 오류 확인
                    response.encoding = response.apparent_encoding # 인코딩 자동 감지 시도
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                        script_or_style.decompose()
                    
                    text = soup.get_text(separator=' ', strip=True)
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    if cleaned_text: # 내용이 있는 경우에만 추가
                        search_results_text.append(cleaned_text[:1500]) # 요약을 위해 조금 더 긴 내용 사용
                    logging.info(f"URL {url} 처리 완료 (내용 길이: {len(cleaned_text)})")
                    
                except requests.exceptions.RequestException as e:
                    logging.warning(f"URL {url} 요청 오류: {str(e)}")
                except Exception as e:
                    logging.warning(f"URL {url} 처리 중 오류: {str(e)}")
                
                if len(search_results_text) >= 2: # 최대 2개의 성공적인 결과만 사용 (요약 부담 줄이기)
                    break
                    
        except Exception as e:
            logging.error(f"구글 검색 API 호출 중 오류: {e}")
            return "웹 검색 중 오류가 발생했습니다."
        
        # 검색 결과가 없으면 바로 반환
        if not search_results_text:
            logging.warning(f"처리 가능한 검색 결과 없음: {query}")
            return "관련 정보를 찾을 수 없습니다."

        # 검색 결과 텍스트들을 하나로 합침
        context = "\n\n---\n\n".join(search_results_text)
        context_for_llm = context[:4000] # LLM 토큰 제한 고려하여 컨텍스트 길이 제한
        logging.info(f"검색 결과 요약 시도 (컨텍스트 길이: {len(context_for_llm)})...")
        try:
            # LLM을 이용한 요약
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Summarize the following text context to directly answer the user's original question. Provide a concise and relevant answer based *only* on the provided text."},
                    {"role": "user", "content": f"Original Question: {query}\n\nContext:\n{context_for_llm}"} # 원본 질문과 컨텍스트 전달
                ],
                temperature=0.3, # 좀 더 사실 기반 요약을 위해 temperature 낮춤
                max_tokens=150 # 요약 길이 제한
            )
            summary = response.choices[0].message.content.strip()
            
            if not summary:
                logging.warning("LLM 요약 결과가 비어 있습니다.")
                # 요약 실패 시, 간단한 결과라도 보여주기 (예: 첫번째 결과 일부)
                return f"검색 결과를 요약하는 데 실패했습니다. 첫 번째 검색 결과 일부: \n{search_results_text[0][:300]}..."

            logging.info("검색 결과 요약 성공")
            return summary
            
        except Exception as e:
            logging.error(f"LLM 요약 API 호출 중 오류: {e}")
            # 요약 실패 시, 간단한 결과라도 보여주기
            return f"검색 결과를 요약하는 중 오류가 발생했습니다. 첫 번째 검색 결과 일부: \n{search_results_text[0][:300]}..."

    def run_task(self, task: str) -> str:
        """주어진 작업을 실행하고 사용자에게 표시할 최종 결과 문자열을 반환"""
        logging.info(f"작업 시작: {task}")
        
        # 키워드 기반 명령어 우선 처리 (파일 관리, 코드 실행 등)
        file_exec_match = re.search(r"실행[:\\s]+([^\s]+)", task)
        compile_match = re.search(r"컴파일[:\\s]+([^\s]+)", task)
        run_match = re.search(r"실행결과[:\\s]+([^\s]+)", task)
        dir_explore_match = re.search(r"탐색[:\\s]+([^\s]+)", task)
        file_manage_match = re.search(r"(생성|삭제|이동)[:\\s]+([^\s]+)(?:\\s+(?:to|->)\\s+([^\s]+))?", task)
        code_match = re.search(r"```(\\w+)?\\n(.*?)```", task, re.DOTALL)
        
        final_result_message = ""

        if file_exec_match:
            # 파일 실행: CodeExecutor 결과 확인 후 처리
            file_path = file_exec_match.group(1)
            execution_result_str = CodeExecutor.execute_file(file_path)
            
            if execution_result_str.startswith("ModuleNotFoundError: "):
                missing_module = execution_result_str.split(": ")[1].strip()
                if missing_module == '_tkinter':
                    final_result_message = f"[오류] Tkinter GUI 라이브러리가 설치되지 않았거나 Python 환경에서 인식되지 않습니다.\nmacOS의 경우 'brew install python-tk' 후 Python 재설치 등을 시도해 보세요."
                else:
                    final_result_message = f"[오류] 코드를 실행하려면 '{missing_module}' 패키지가 필요합니다.\n터미널에서 'pip install {missing_module}' 명령어로 설치해주세요."
            elif execution_result_str.startswith("FileNotFoundError: Required command "):
                missing_command = re.search(r"'(.+?)'", execution_result_str).group(1)
                final_result_message = f"[오류] 코드 실행에 필요한 '{missing_command}' 명령어를 찾을 수 없습니다.\n관련 언어/도구를 설치하고 PATH 환경 변수를 확인해주세요."
            else:
                final_result_message = execution_result_str
        
        elif compile_match:
            # 파일 컴파일
            file_path = compile_match.group(1)
            _, language = FileManager.analyze_file(file_path)
            
            if language not in ['c++', 'c', 'rust', 'c#']:
                final_result_message = f"컴파일이 필요하지 않은 언어입니다: {language}"
            else:
                temp_dir = CodeExecutor.get_temp_dir()
                output_file_no_ext = os.path.join(temp_dir, 'temp_out')
                output_file_with_ext = output_file_no_ext
                if language == 'c#': output_file_with_ext += ".exe"
                elif language in ['c++', 'c', 'rust']: output_file_with_ext = output_file_no_ext # 확장자 없음
                
                cmd = list(CodeExecutor.COMMAND_MAP[language])
                compile_cmd = cmd + [file_path]
                if '{output}' in cmd:
                    idx = cmd.index('{output}')
                    if language == 'c#': compile_cmd[idx] = f'/out:{output_file_with_ext}'
                    else: compile_cmd[idx] = output_file_with_ext
                
                compile_ret, _, compile_stderr = CodeExecutor._execute_with_popen(compile_cmd, timeout=30)
                if compile_ret == 0:
                    final_result_message = "컴파일이 완료되었습니다."
                    logging.info("컴파일 성공")
                else:
                    final_result_message = f"컴파일 중 오류 발생:\n{compile_stderr}"
                    logging.error(f"컴파일 실패: {compile_stderr}")
        
        elif run_match:
            # 컴파일된 파일 실행
            file_path = run_match.group(1)
            _, language = FileManager.analyze_file(file_path) # 언어 분석 추가
            temp_dir = CodeExecutor.get_temp_dir()
            output_file_no_ext = os.path.join(temp_dir, 'temp_out')
            output_file_with_ext = output_file_no_ext
            if language == 'c#': output_file_with_ext += ".exe"
            
            if not os.path.exists(output_file_with_ext):
                final_result_message = f"{file_path}에 대한 컴파일된 실행 파일({output_file_with_ext})이 없습니다. 먼저 컴파일해주세요."
            else:
                cmd_to_run = [output_file_with_ext]
                if language == 'c#' and os.name != 'nt':
                    cmd_to_run.insert(0, 'mono') # mono 추가
                
                run_ret, run_stdout, run_stderr = CodeExecutor._execute_with_popen(cmd_to_run, timeout=10)
                
                module_match = re.search(r"ModuleNotFoundError: No module named '(.+?)'", run_stderr) # 실행 결과에서도 확인
                file_not_found_match = re.match(r"FileNotFoundError: Required command '(.+?)' not found", run_stderr)

                if module_match:
                    missing_module = module_match.group(1)
                    if missing_module == '_tkinter':
                        final_result_message = "[오류] Tkinter GUI 라이브러리가 설치되지 않았거나 인식되지 않습니다."
                    else:
                        final_result_message = f"[오류] '{missing_module}' 패키지가 필요합니다. 'pip install {missing_module}' 명령어로 설치해주세요."
                elif file_not_found_match:
                    missing_command = re.search(r"'(.+?)'", run_stderr).group(1)
                    final_result_message = f"[오류] '{missing_command}' 명령어를 찾을 수 없습니다. 관련 언어/도구를 설치해주세요."
                elif run_ret == 0:
                    final_result_message = f"실행 결과:\n{run_stdout}"
                    if run_stderr:
                        final_result_message += f"\nStandard Error:\n{run_stderr}"
                else:
                    final_result_message = f"실행 중 오류 발생:\n{run_stderr}"
                    if run_stdout:
                        final_result_message += f"\nStandard Output:\n{run_stdout}"
        
        elif dir_explore_match:
            # 디렉토리 탐색
            dir_path = dir_explore_match.group(1)
            explore_result = FileManager.explore_directory(dir_path)
            
            if explore_result:
                final_result_message = f"디렉토리: {explore_result['path']}\n"
                final_result_message += f"총 크기: {explore_result['total_size']:,} bytes\n"
                final_result_message += f"항목 수: {len(explore_result['items'])}\n\n"
                
                final_result_message += "파일 목록:\n"
                for item in explore_result['items']:
                    if item['type'] == 'file':
                        final_result_message += f"📄 {item['name']} ({item['size']:,} bytes) - {item['file_type']} [{item['language']}]\n"
                    else:
                        final_result_message += f"📁 {item['name']} ({item['size']:,} bytes)\n"
            else:
                final_result_message = "디렉토리 탐색 중 오류가 발생했습니다."
        
        elif file_manage_match:
            # 파일 관리
            action_map = {'생성': 'create', '삭제': 'delete', '이동': 'move'}
            action = action_map[file_manage_match.group(1)]
            path = file_manage_match.group(2)
            new_path = file_manage_match.group(3) if action == 'move' else None
            final_result_message = FileManager.manage_files(action, path, new_path)
        
        elif code_match:
            # 코드 블록 실행: CodeExecutor 결과 확인 후 처리
            language = code_match.group(1) or "python"
            code = code_match.group(2)
            execution_result_str = CodeExecutor.execute_code(code, language)
            
            if execution_result_str.startswith("ModuleNotFoundError: "):
                missing_module = execution_result_str.split(": ")[1].strip()
                if missing_module == '_tkinter':
                    final_result_message = f"[오류] Tkinter GUI 라이브러리가 설치되지 않았거나 Python 환경에서 인식되지 않습니다.\nmacOS의 경우 'brew install python-tk' 후 Python 재설치 등을 시도해 보세요."
                else:
                    final_result_message = f"[오류] 코드를 실행하려면 '{missing_module}' 패키지가 필요합니다.\n터미널에서 'pip install {missing_module}' 명령어로 설치해주세요."
            elif execution_result_str.startswith("FileNotFoundError: Required command "):
                missing_command = re.search(r"'(.+?)'", execution_result_str).group(1)
                final_result_message = f"[오류] 코드 실행에 필요한 '{missing_command}' 명령어를 찾을 수 없습니다.\n관련 언어/도구를 설치하고 PATH 환경 변수를 확인해주세요."
            else:
                final_result_message = execution_result_str
        
        else:
            # ToolCallingAgent 처리 (코드 생성 + 자동 설치/실행 + 검색)
            agent_result = self.agent.run(task, print_results=False)
            result_type = agent_result.get('result_type')
            
            if result_type == 'code_generation':
                # 코드 생성 결과 처리
                generated_code = agent_result.get('generated_code', '코드 내용을 가져올 수 없습니다.')
                save_msg = agent_result.get('saved_path_message', '저장 경로 정보 없음')
                saved_file_path = agent_result.get('saved_file_path')
                execute_request = agent_result.get('execute_request', False)
                language_name = agent_result.get('language', 'unknown')
                required_packages = agent_result.get('required_packages', [])
                
                # 사용자에게 보여줄 기본 메시지 (코드 + 저장 경로)
                final_result_message = f"--- 생성된 코드 ({language_name}) ---\n{generated_code}\n-------------------\n{save_msg}\""
                
                installation_successful = True # 기본값: 설치 필요 없거나 성공
                if required_packages:
                    # 설치 시도 로깅 추가
                    logging.info(f"필요 패키지 감지됨: {required_packages}. 자동 설치 시도...")
                    install_command_str = f"{sys.executable} -m pip install {' '.join(required_packages)}\""
                    final_result_message += f"\n\n[알림] 다음 패키지 자동 설치 시도: {', '.join(required_packages)}\""
                    
                    # pip install 실행 (sys.executable 사용)
                    install_command_list = [sys.executable, "-m", "pip", "install"] + required_packages
                    install_ret, install_stdout, install_stderr = CodeExecutor._execute_with_popen(install_command_list, timeout=120)
                    
                    if install_ret == 0:
                        logging.info(f"패키지 설치 성공: {', '.join(required_packages)}\"")
                        final_result_message += "\n설치 성공."
                    else:
                        installation_successful = False
                        logging.error(f"패키지 설치 실패: {install_stderr}\"")
                        final_result_message += f"\n설치 실패:\n{install_stderr[:500]}..."
                        if execute_request:
                            final_result_message += "\n(패키지 설치 실패로 자동 실행할 수 없습니다.)"
                            execute_request = False
                
                # 실행 요청 처리 (CodeExecutor 결과 확인 후 처리)
                if execute_request and installation_successful and saved_file_path:
                    logging.info(f"생성된 코드 파일 실행 시도: {saved_file_path}")
                    execution_result_str = CodeExecutor.execute_file(saved_file_path)
                    # 실행 결과에 따라 메시지 분기
                    if execution_result_str.startswith("ModuleNotFoundError: "):
                        missing_module = execution_result_str.split(": ")[1].strip()
                        if missing_module == '_tkinter':
                            final_result_message += f"\n\n--- 코드 실행 결과 ---\n[오류] Tkinter GUI 라이브러리가 설치되지 않았거나 Python 환경에서 인식되지 않습니다.\nmacOS의 경우 'brew install python-tk' 후 Python 재설치 등을 시도해 보세요."
                        else:
                            final_result_message += f"\n\n--- 코드 실행 결과 ---\n[오류] 코드를 실행하려면 '{missing_module}' 패키지가 필요합니다.\n터미널에서 'pip install {missing_module}' 명령어로 설치해주세요."
                    elif execution_result_str.startswith("FileNotFoundError: Required command "):
                        missing_command = re.search(r"'(.+?)'", execution_result_str).group(1)
                        final_result_message += f"\n\n--- 코드 실행 결과 ---\n[오류] 코드 실행에 필요한 '{missing_command}' 명령어를 찾을 수 없습니다.\n관련 언어/도구를 설치하고 PATH 환경 변수를 확인해주세요."
                    else:
                        final_result_message += f"\n\n--- 코드 실행 결과 ---\n{execution_result_str}\n---------------------"
                elif execute_request and not saved_file_path: # 저장 실패 시
                    final_result_message += "\n\n(파일 저장 실패로 실행할 수 없습니다.)"
            
            elif result_type == 'error':
                final_result_message = agent_result.get('result', 'LLM 처리 중 알 수 없는 오류 발생')
            else: # 'unknown'
                logging.info(f"처리할 명령어가 없어 웹 검색 시도: {task}")
                final_result_message = self._search(task)

        # 최종 결과 메시지를 메모리에 저장
        self.memory.append({
            "task": task,
            "result": final_result_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # 메모리 관리
        self._manage_memory()
        
        logging.info("작업 완료")
        return final_result_message

if __name__ == "__main__":
    # 에이전트 생성
    agent = AgentAI(
        name="도우미",
        description="저는 여러분의 질문에 답하고 작업을 도와주는 AI 에이전트입니다.",
        memory_limit=5
    )
    
    # 대화형 모드로 실행
    agent.run_interactive() 