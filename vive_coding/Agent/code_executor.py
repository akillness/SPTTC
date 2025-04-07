import os
import logging
import subprocess
import threading
import errno
import re
from typing import Dict, List, Tuple
from file_manager import FileManager

class CodeExecutor:
    """코드 실행 클래스"""
    
    # 언어별 실행 명령어 매핑
    COMMAND_MAP = {
        'python': ['python3'],
        'javascript': ['node'],
        'java': ['java'],
        'c++': ['g++', '-o', '{output}', '-std=c++11'],
        'c': ['gcc', '-o', '{output}'],
        'c#': ['csc', '/out:{output}'],
        'go': ['go', 'run'],
        'rust': ['rustc', '-o', '{output}'],
        'ruby': ['ruby'],
        'php': ['php'],
        'r': ['Rscript']
    }
    
    @staticmethod
    def get_temp_dir() -> str:
        """임시 파일을 저장할 디렉토리 경로 반환"""
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    @staticmethod
    def _execute_with_popen(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
        """Popen을 사용하여 명령어를 실행하고 결과를 반환"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
            
        except FileNotFoundError as e:
            # 실행 파일 (python3, go, csc 등) 못찾는 경우
            if e.errno == errno.ENOENT:
                executable_name = cmd[0]
                logging.error(f"Executable not found: {executable_name}. Error: {e}")
                return -2, "", f"FileNotFoundError: Required command '{executable_name}' not found. Please install the corresponding language/tool and ensure it's in the system PATH."
            else:
                 logging.error(f"File not found error during Popen execution: {e}")
                 return -1, "", f"Execution Error: {str(e)}"
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            logging.warning(f"Process timed out after {timeout} seconds.")
            return -1, stdout, f"Timeout Error: Process exceeded {timeout} seconds.\n{stderr}"
        except Exception as e:
            process.kill()
            logging.error(f"Error during Popen execution: {e}")
            return -1, stdout, f"Execution Error: {str(e)}\n{stderr}"
            
        return returncode, stdout, stderr

    @staticmethod
    def execute_code(code: str, language: str = "python") -> str:
        """코드 문자열을 실행 (Popen 사용, ModuleNotFoundError 감지)"""
        logging.info(f"코드 실행 시도 (언어: {language}) using Popen")
        
        if language not in CodeExecutor.COMMAND_MAP:
            return f"지원하지 않는 프로그래밍 언어입니다: {language}"
        
        temp_dir = CodeExecutor.get_temp_dir()
        temp_file = os.path.join(temp_dir, f'temp_code.{language}')
        output_file = os.path.join(temp_dir, 'temp_out')
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        cmd = list(CodeExecutor.COMMAND_MAP[language])
        needs_compile = language in ['c++', 'c', 'rust', 'c#']
        
        compile_success = True
        compile_stderr = ""
        output_file_with_ext = output_file
        if language == 'c#':
             output_file_with_ext += ".exe"

        if needs_compile:
            compile_cmd = cmd + [temp_file]
            if '{output}' in cmd:
                idx = cmd.index('{output}')
                if language == 'c#':
                    compile_cmd[idx] = f'/out:{output_file_with_ext}'
                else:
                    compile_cmd[idx] = output_file_with_ext
                
            compile_ret, _, compile_stderr = CodeExecutor._execute_with_popen(compile_cmd, timeout=30)
            if compile_ret != 0:
                compile_success = False
            else:
                 cmd = [output_file_with_ext]
        else:
            cmd.append(temp_file)

        result_str = ""
        if not compile_success:
             result_str = f"컴파일 중 오류 발생:\n{compile_stderr}"
             logging.error(f"컴파일 실패: {compile_stderr}")
        else:
            if language == 'c#' and os.name != 'nt':
                 cmd.insert(0, 'mono')
            
            returncode, stdout, stderr = CodeExecutor._execute_with_popen(cmd, timeout=10)
            
            # ModuleNotFoundError 감지
            module_match = re.search(r"ModuleNotFoundError: No module named '(.+?)'", stderr)
            file_not_found_match = re.match(r"FileNotFoundError: Required command '(.+?)' not found", stderr)

            if module_match:
                 missing_module = module_match.group(1)
                 logging.warning(f"Module not found: {missing_module}")
                 result_str = f"ModuleNotFoundError: {missing_module}"
            elif file_not_found_match:
                 missing_command = file_not_found_match.group(1)
                 logging.error(f"Command not found during execution: {missing_command}")
                 result_str = stderr
            elif returncode == 0:
                logging.info("코드 실행 성공 (Popen)")
                result_str = f"실행 결과:\n{stdout}"
                if stderr:
                    result_str += f"\nStandard Error:\n{stderr}"
            elif returncode == -1 and "Timeout Error" in stderr:
                 result_str = f"실행 시간 초과 (10초):\n{stdout}\n{stderr}"
                 logging.warning(f"코드 실행 시간 초과: {stderr}")
            else:
                logging.error(f"코드 실행 실패 (Popen): {stderr}")
                result_str = f"실행 중 오류 발생:\n{stderr}"
                if stdout:
                     result_str += f"\nStandard Output:\n{stdout}"

        # 임시 파일 정리
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(output_file_with_ext):
                os.unlink(output_file_with_ext)
        except OSError as e:
             logging.warning(f"Error deleting temp files: {e}")
             
        return result_str

    @staticmethod
    def execute_file(file_path: str) -> str:
        """파일을 실행 (Popen 사용, ModuleNotFoundError 감지)"""
        logging.info(f"파일 실행 시도: {file_path} using Popen")
        
        if not os.path.exists(file_path):
            return f"파일을 찾을 수 없습니다: {file_path}"
        
        _, language = FileManager.analyze_file(file_path)
        
        if language not in CodeExecutor.COMMAND_MAP:
            return f"지원하지 않는 프로그래밍 언어입니다: {language}"
            
        temp_dir = CodeExecutor.get_temp_dir()
        output_file_no_ext = os.path.join(temp_dir, 'temp_out')
        cmd = list(CodeExecutor.COMMAND_MAP[language])
        needs_compile = language in ['c++', 'c', 'rust', 'c#']

        compile_success = True
        compile_stderr = ""
        output_file_with_ext = output_file_no_ext
        if language == 'c#':
            output_file_with_ext += ".exe"

        if needs_compile:
            compile_cmd = cmd + [file_path]
            if '{output}' in cmd:
                idx = cmd.index('{output}')
                if language == 'c#':
                    compile_cmd[idx] = f'/out:{output_file_with_ext}'
                else:
                    compile_cmd[idx] = output_file_with_ext

            compile_ret, _, compile_stderr = CodeExecutor._execute_with_popen(compile_cmd, timeout=30)
            if compile_ret != 0:
                compile_success = False
            else:
                cmd = [output_file_with_ext]
        else:
            cmd.append(file_path)

        result_str = ""
        if not compile_success:
             result_str = f"컴파일 중 오류 발생:\n{compile_stderr}"
             logging.error(f"컴파일 실패: {compile_stderr}")
        else:
            if language == 'c#' and os.name != 'nt':
                 cmd.insert(0, 'mono')
                 
            returncode, stdout, stderr = CodeExecutor._execute_with_popen(cmd, timeout=10)
            
            # ModuleNotFoundError 감지
            module_match = re.search(r"ModuleNotFoundError: No module named '(.+?)'", stderr)
            file_not_found_match = re.match(r"FileNotFoundError: Required command '(.+?)' not found", stderr)

            if module_match:
                 missing_module = module_match.group(1)
                 logging.warning(f"Module not found: {missing_module}")
                 result_str = f"ModuleNotFoundError: {missing_module}"
            elif file_not_found_match:
                 missing_command = file_not_found_match.group(1)
                 logging.error(f"Command not found during execution: {missing_command}")
                 result_str = stderr
            elif returncode == 0:
                logging.info("파일 실행 성공 (Popen)")
                result_str = f"실행 결과:\n{stdout}"
                if stderr:
                    result_str += f"\nStandard Error:\n{stderr}"
            elif returncode == -1 and "Timeout Error" in stderr:
                 result_str = f"실행 시간 초과 (10초):\n{stdout}\n{stderr}"
                 logging.warning(f"파일 실행 시간 초과: {stderr}")
            else:
                logging.error(f"파일 실행 실패 (Popen): {stderr}")
                result_str = f"실행 중 오류 발생:\n{stderr}"
                if stdout:
                     result_str += f"\nStandard Output:\n{stdout}"

        # 임시 파일 정리 (컴파일 출력물)
        try:
            if os.path.exists(output_file_with_ext):
                os.unlink(output_file_with_ext)
        except OSError as e:
             logging.warning(f"Error deleting temp output file: {e}")

        return result_str 