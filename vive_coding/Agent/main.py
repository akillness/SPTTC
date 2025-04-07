from typing import List, Dict, Any, Callable, Optional, Tuple
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
        
        # 에이전트 및 도구 초기화
        self.code_generator = CodeGeneratorAgent(client=self.client) # CodeGeneratorAgent 인스턴스 생성
        self._initialize_task_handlers() # 작업 핸들러 초기화
        
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

    # --- 작업 처리 로직 --- 
    def _initialize_task_handlers(self):
        """명령어 패턴과 처리 함수를 매핑"""
        self.task_handlers: List[Tuple[re.Pattern, Callable]] = [
            # 파일 실행: "실행 파일명"
            (re.compile(r"^실행\s+(.+)"), self._handle_file_execution),
            # 코드 블록 실행: ``` ```
            (re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL), self._handle_code_block_execution),
            # 컴파일: "컴파일 파일명"
            (re.compile(r"^컴파일\s+(.+)"), self._handle_compilation),
            # 컴파일 결과 실행: "실행결과 원본파일명"
            (re.compile(r"^실행결과\s+(.+)"), self._handle_compiled_run),
            # 디렉토리 탐색: "탐색 경로"
            (re.compile(r"^탐색\s+(.+)"), self._handle_directory_exploration),
            # 파일 관리: "생성/삭제/이동 경로 [to 새경로]"
            (re.compile(r"^(생성|삭제|이동)\s+([^\s]+)(?:\s+(?:to|->)\s+([^\s]+))?"), self._handle_file_management),
            # 코드 생성 (검색 키워드 포함 가능): "검색해서/검색하여 [...] 코드/프로그램 [...] 실행[해줘]"
            (re.compile(r"(검색(?:해서|하여)\s+)?.*(?:코드|프로그램|작성|만들|짜줘|generate|create|write|code|program).*(실행|돌려|run|execute)?", re.IGNORECASE), self._handle_code_generation),
            # 기본 처리 (웹 검색)
            (re.compile(r".*"), self._handle_default) # 가장 마지막에 위치해야 함
        ]
        
    def _format_execution_result(self, execution_result_str: str) -> str:
        """CodeExecutor 결과를 사용자 친화적 메시지로 포맷"""
        if execution_result_str.startswith("ModuleNotFoundError: "):
            missing_module = execution_result_str.split(": ")[1].strip()
            if missing_module == '_tkinter':
                return f"[오류] Tkinter GUI 라이브러리가 설치되지 않았거나 Python 환경에서 인식되지 않습니다.\nmacOS의 경우 'brew install python-tk' 후 Python 재설치 등을 시도해 보세요."
            else:
                return f"[오류] 코드를 실행하려면 '{missing_module}' 패키지가 필요합니다.\n터미널에서 'pip install {missing_module}' 명령어로 설치해주세요."
        elif execution_result_str.startswith("FileNotFoundError: Required command "):
            missing_command = re.search(r"'(.+?)'", execution_result_str).group(1)
            return f"[오류] 코드 실행에 필요한 '{missing_command}' 명령어를 찾을 수 없습니다.\n관련 언어/도구를 설치하고 PATH 환경 변수를 확인해주세요."
        else:
            return execution_result_str # 오류 없으면 그대로 반환

    def _handle_file_execution(self, match: re.Match) -> str:
        """파일 실행 처리"""
        file_path = match.group(1)
        result_str = CodeExecutor.execute_file(file_path)
        return self._format_execution_result(result_str)

    def _handle_code_block_execution(self, match: re.Match) -> str:
        """코드 블록 실행 처리"""
        language = match.group(1) or "python"
        code = match.group(2)
        result_str = CodeExecutor.execute_code(code, language)
        return self._format_execution_result(result_str)

    def _handle_compilation(self, match: re.Match) -> str:
        """파일 컴파일 처리"""
        file_path = match.group(1)
        _, language = FileManager.analyze_file(file_path)
        
        if language not in ['c++', 'c', 'rust', 'c#']:
            return f"컴파일이 필요하지 않은 언어입니다: {language}"
        
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
            logging.info("컴파일 성공")
            return "컴파일이 완료되었습니다."
        else:
            logging.error(f"컴파일 실패: {compile_stderr}")
            return f"컴파일 중 오류 발생:\n{compile_stderr}"

    def _handle_compiled_run(self, match: re.Match) -> str:
        """컴파일된 파일 실행 처리"""
        file_path = match.group(1) # 원본 파일 경로
        _, language = FileManager.analyze_file(file_path)
        temp_dir = CodeExecutor.get_temp_dir()
        output_file_no_ext = os.path.join(temp_dir, 'temp_out')
        output_file_with_ext = output_file_no_ext
        if language == 'c#': output_file_with_ext += ".exe"
        
        if not os.path.exists(output_file_with_ext):
            return f"{file_path}에 대한 컴파일된 실행 파일({output_file_with_ext})이 없습니다. 먼저 컴파일해주세요."
        
        cmd_to_run = [output_file_with_ext]
        if language == 'c#' and os.name != 'nt':
            cmd_to_run.insert(0, 'mono')
        
        run_ret, run_stdout, run_stderr = CodeExecutor._execute_with_popen(cmd_to_run, timeout=10)
        
        # 실행 결과 포맷팅 (오류 포함)
        combined_output = f"실행 결과:\n{run_stdout}" if run_stdout else ""
        if run_stderr:
            # ModuleNotFoundError, FileNotFoundError 등 특정 오류 처리
            formatted_stderr = self._format_execution_result(run_stderr)
            if formatted_stderr == run_stderr: # 포맷팅 안된 일반 오류
                 combined_output += f"\nStandard Error:\n{formatted_stderr}"
            else: # 포맷팅된 특정 오류 메시지
                 combined_output = formatted_stderr # 오류 메시지만 반환
        
        if run_ret != 0 and not any(err in combined_output for err in ["[오류]", "ModuleNotFoundError", "FileNotFoundError"]):
            # 명시적 오류 메시지가 없고 종료 코드가 0이 아니면 일반 오류로 처리
            final_message = f"실행 중 오류 발생 (종료 코드: {run_ret}):\n{combined_output}"
        else:
            final_message = combined_output
            
        return final_message

    def _handle_directory_exploration(self, match: re.Match) -> str:
        """디렉토리 탐색 처리"""
        dir_path = match.group(1)
        explore_result = FileManager.explore_directory(dir_path)
        
        if explore_result:
            message = f"디렉토리: {explore_result['path']}\n"
            message += f"총 크기: {explore_result['total_size']:,} bytes\n"
            message += f"항목 수: {len(explore_result['items'])}\n\n"
            message += "파일 목록:\n"
            for item in explore_result['items']:
                if item['type'] == 'file':
                    message += f"📄 {item['name']} ({item['size']:,} bytes) - {item['file_type']} [{item['language']}]\n"
                else:
                    message += f"📁 {item['name']} ({item['size']:,} bytes)\n"
            return message
        else:
            return "디렉토리 탐색 중 오류가 발생했습니다."

    def _handle_file_management(self, match: re.Match) -> str:
        """파일 관리 처리"""
        action_map = {'생성': 'create', '삭제': 'delete', '이동': 'move'}
        action = action_map[match.group(1)]
        path = match.group(2)
        new_path = match.group(3) if action == 'move' else None
        return FileManager.manage_files(action, path, new_path)

    def _handle_code_generation(self, match: re.Match) -> str:
        """코드 생성 처리 (검색 결과 활용 및 자동 실행 포함)"""
        task = match.group(0) # 매칭된 전체 문자열을 task로 사용
        needs_search = bool(match.group(1)) # "검색해서" 등이 있는지 확인
        
        search_context = None
        if needs_search:
            # '검색해서' 부분 제외하고 실제 검색 쿼리 추출 (간단한 방식)            
            search_query = task.replace(match.group(1), "").strip()
            logging.info(f"코드 생성을 위해 웹 검색 수행: {search_query}")
            search_context = self._search(search_query)
            if "오류" in search_context or "찾을 수 없습니다" in search_context:
                logging.warning(f"검색 실패 또는 결과 없음: {search_context}")
                # 검색 실패해도 코드 생성은 시도
                # return f"코드 생성에 필요한 정보를 검색하지 못했습니다: {search_context}"
        
        # CodeGeneratorAgent 호출 (검색 결과 전달)
        agent_result = self.code_generator.run(task, search_context=search_context)
        result_type = agent_result.get('result_type')
        
        final_message = ""
        
        if result_type == 'code_generation':
            generated_code = agent_result.get('generated_code', '코드 내용을 가져올 수 없습니다.')
            save_msg = agent_result.get('saved_path_message', '저장 경로 정보 없음')
            saved_file_path = agent_result.get('saved_file_path')
            execute_request = agent_result.get('execute_request', False)
            language_name = agent_result.get('language', 'unknown')
            required_packages = agent_result.get('required_packages', [])
            
            final_message = f"--- 생성된 코드 ({language_name}) ---\n{generated_code}\n-------------------\n{save_msg}"
            
            installation_successful = True
            if required_packages:
                logging.info(f"필요 패키지 감지됨: {required_packages}. 자동 설치 시도...")
                final_message += f"\n\n[알림] 다음 패키지 자동 설치 시도: {', '.join(required_packages)}"
                
                install_command_list = [sys.executable, "-m", "pip", "install"] + required_packages
                install_ret, _, install_stderr = CodeExecutor._execute_with_popen(install_command_list, timeout=120)
                
                if install_ret == 0:
                    logging.info(f"패키지 설치 성공: {', '.join(required_packages)}")
                    final_message += "\n설치 성공."
                else:
                    installation_successful = False
                    logging.error(f"패키지 설치 실패: {install_stderr}")
                    final_message += f"\n설치 실패:\n{install_stderr[:500]}..."
                    if execute_request:
                        final_message += "\n(패키지 설치 실패로 자동 실행할 수 없습니다.)"
                        execute_request = False
            
            if execute_request and installation_successful and saved_file_path:
                logging.info(f"생성된 코드 파일 실행 시도: {saved_file_path}")
                execution_result_str = CodeExecutor.execute_file(saved_file_path)
                formatted_execution_result = self._format_execution_result(execution_result_str)
                final_message += f"\n\n--- 코드 실행 결과 ---\n{formatted_execution_result}\n---------------------"
            elif execute_request and not saved_file_path:
                final_message += "\n\n(파일 저장 실패로 실행할 수 없습니다.)"
                
        elif result_type == 'error':
            final_message = agent_result.get('result', 'LLM 처리 중 알 수 없는 오류 발생')
        else: # 'unknown' 또는 다른 예상치 못한 타입
             final_message = agent_result.get('result', '코드 생성 에이전트가 작업을 처리하지 못했습니다.')
             
        return final_message

    def _handle_default(self, match: re.Match) -> str:
        """기본 처리 (웹 검색) """
        task = match.group(0)
        logging.info(f"처리할 특정 명령어가 없어 웹 검색 시도: {task}")
        return self._search(task)

    def run_task(self, task: str) -> str:
        """주어진 작업을 적절한 핸들러에 전달하여 실행"""
        logging.info(f"작업 시작: {task}")
        
        final_result_message = "알 수 없는 오류가 발생했습니다." # 기본 오류 메시지
        
        handler_found = False
        for pattern, handler in self.task_handlers:
            match = pattern.match(task)
            if match:
                try:
                    logging.info(f"핸들러 {handler.__name__} 선택됨")
                    final_result_message = handler(match)
                    handler_found = True
                    break # 첫 번째 매칭되는 핸들러 사용
                except Exception as e:
                    logging.error(f"{handler.__name__} 실행 중 오류: {e}", exc_info=True)
                    final_result_message = f"{handler.__name__} 처리 중 오류가 발생했습니다: {e}"
                    break # 오류 발생 시 중단
                    
        if not handler_found:
             # 이 경우는 _handle_default 패턴이 항상 매칭되므로 발생하지 않아야 함
             logging.warning("처리할 핸들러를 찾지 못했습니다. 기본 검색 핸들러가 제대로 설정되었는지 확인하세요.")
             final_result_message = "요청을 처리할 방법을 찾지 못했습니다."

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