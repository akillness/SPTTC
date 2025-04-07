import os
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Tuple, Dict, List

class FileManager:
    """파일 관리 클래스"""
    
    # 개발 언어 매핑
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.java': 'java',
        '.cpp': 'c++',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'c#',
        '.ts': 'typescript',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.r': 'r',
        '.m': 'matlab',
        '.scala': 'scala'
    }
    
    @staticmethod
    def analyze_file(file_path: str) -> Tuple[str, str]:
        """파일 분석하여 파일 타입과 개발 언어를 파악"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            # 파일 확장자 확인
            ext = os.path.splitext(file_path)[1].lower()
            
            # MIME 타입 확인
            mime_type, _ = mimetypes.guess_type(file_path)
            
            language = FileManager.LANGUAGE_MAP.get(ext, 'unknown')
            file_type = mime_type if mime_type else f"application/{ext[1:]}"
            
            logging.info(f"파일 분석 완료 - 타입: {file_type}, 언어: {language}")
            return file_type, language
            
        except Exception as e:
            error_msg = f"파일 분석 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return 'unknown', 'unknown'
    
    @staticmethod
    def explore_directory(dir_path: str = '.') -> Dict:
        """디렉토리 내용을 탐색"""
        try:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {dir_path}")
            
            items = []
            total_size = 0
            
            for item in os.scandir(dir_path):
                try:
                    stats = item.stat()
                    size = stats.st_size
                    total_size += size
                    
                    if item.is_file():
                        file_type, language = FileManager.analyze_file(item.path)
                        items.append({
                            'name': item.name,
                            'type': 'file',
                            'size': size,
                            'file_type': file_type,
                            'language': language
                        })
                    else:
                        items.append({
                            'name': item.name,
                            'type': 'directory',
                            'size': size
                        })
                except Exception as e:
                    logging.warning(f"항목 처리 중 오류: {str(e)}")
                    continue
            
            return {
                'path': os.path.abspath(dir_path),
                'total_size': total_size,
                'items': sorted(items, key=lambda x: x['name'])
            }
            
        except Exception as e:
            error_msg = f"디렉토리 탐색 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return {}
    
    @staticmethod
    def manage_files(action: str, path: str, new_path: str = None) -> str:
        """파일 관리 (생성, 삭제, 이동)"""
        try:
            if action == 'create':
                if os.path.exists(path):
                    return f"이미 존재합니다: {path}"
                
                if path.endswith('/'):
                    os.makedirs(path, exist_ok=True)
                    logging.info(f"디렉토리 생성 완료: {path}")
                    return f"디렉토리가 생성되었습니다: {path}"
                else:
                    Path(path).touch()
                    logging.info(f"파일 생성 완료: {path}")
                    return f"파일이 생성되었습니다: {path}"
                    
            elif action == 'delete':
                if not os.path.exists(path):
                    return f"존재하지 않습니다: {path}"
                
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logging.info(f"디렉토리 삭제 완료: {path}")
                    return f"디렉토리가 삭제되었습니다: {path}"
                else:
                    os.remove(path)
                    logging.info(f"파일 삭제 완료: {path}")
                    return f"파일이 삭제되었습니다: {path}"
                    
            elif action == 'move':
                if not os.path.exists(path):
                    return f"원본이 존재하지 않습니다: {path}"
                if os.path.exists(new_path):
                    return f"대상이 이미 존재합니다: {new_path}"
                
                shutil.move(path, new_path)
                logging.info(f"파일/디렉토리 이동 완료: {path} -> {new_path}")
                return f"이동되었습니다: {path} -> {new_path}"
                
            else:
                return f"지원하지 않는 작업입니다: {action}"
                
        except Exception as e:
            error_msg = f"파일 관리 중 오류 발생: {str(e)}"
            logging.error(error_msg)
            return error_msg 