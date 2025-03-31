import os
from typing import List, Dict, Tuple, Set, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import platform
import matplotlib as mpl
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
elif platform.system() == 'Windows':  # Windows
    font_path = 'C:/Windows/Fonts/malgun.ttf'
else:  # Linux
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

# 폰트 패스가 존재하는지 확인하고 설정
if Path(font_path).exists():
    plt.rcParams['font.family'] = 'AppleGothic' if platform.system() == 'Darwin' else 'NanumGothic'
    mpl.font_manager.fontManager.addfont(font_path)
    plt.rcParams['axes.unicode_minus'] = False
    print(f"✅ 폰트 설정 완료: {font_path}")
else:
    print("⚠️ 한글 폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# 임베딩 모델 초기화
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

class ResumeSummarizer:
    def __init__(self):
        # 초기화 로직 간소화
        self._initialize_settings()
        self._setup_configurations()
        
    def _initialize_settings(self):
        """공통 설정 초기화"""
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.node_parser = SentenceSplitter(
            chunk_size=512, chunk_overlap=50, paragraph_separator="\n"
        )
        self.pdf_directory = self._get_pdf_directory()

    def _get_pdf_directory(self) -> str:
        """PDF 디렉토리 경로 반환"""
        pdf_dir = 'data'
        if not os.path.exists(pdf_dir):
            pdf_dir = os.path.join(os.path.dirname(__file__), 'data')
            if not os.path.exists(pdf_dir):
                os.makedirs(pdf_dir)
        return pdf_dir

    def _setup_configurations(self):
        """구성 요소 설정"""
        self.tech_categories = {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
                "swift", "kotlin", "php", "ruby", "scala", "r", "matlab"
            ],
            "frameworks": [
                "django", "flask", "fastapi", "spring", "react", "vue", "angular",
                "node.js", "express", "tensorflow", "pytorch", "keras", "scikit-learn"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
                "oracle", "sql server", "sqlite", "mariadb"
            ],
            "cloud_devops": [
                "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab",
                "terraform", "ansible", "prometheus", "grafana"
            ],
            "ai_ml": [
                "machine learning", "deep learning", "nlp", "computer vision",
                "reinforcement learning", "neural networks", "transformers", "llm"
            ]
        }
        
        self.section_keywords = {
            "personal_info": ["objective", "contact", "email", "phone", "address", "자기소개", "이메일", "연락처", "profile"],
            "education": ["education", "university", "degree", "school", "학교", "학위", "전공", "교육", "academic"],
            "experience": ["experience", "work", "career", "company", "회사", "경력", "직장", "근무", "employment"],
            "skills": ["skills", "technologies", "programming", "tools", "기술", "프로그래밍", "도구", "역량", "tech stack"],
            "projects": ["project", "developed", "implemented", "프로젝트", "개발", "구현", "설계", "deployed"],
            "publications": ["publication", "paper", "conference", "journal", "논문", "학회", "저널", "research"],
            "awards": ["award", "achievement", "honor", "수상", "성과", "실적", "recognition"],
            "certifications": ["certification", "license", "certificate", "자격증", "인증", "qualified"],
        }
        
        self.date_patterns = [
            r'\d{4}[-./년]\s*\d{1,2}[-./월]?',  # YYYY-MM 또는 YYYY년 MM월
            r'\d{1,2}[-./월]\s*\d{4}[-./년]?',  # MM-YYYY 또는 MM월 YYYY년
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s.,-]+\d{4}',  # Month YYYY
            r'\d{4}\s*[-~–]\s*\d{4}',  # YYYY-YYYY (범위)
            r'\d{4}\s*[-~–]\s*(현재|present|now)',  # YYYY-현재
            r'\d{4}',  # YYYY
            r'현재|present|now'  # 현재 시점
        ]

        self.date_range_patterns = [
            r'(\d{4}[-./]\d{1,2})\s*[-~–]\s*(\d{4}[-./]\d{1,2})',  # YYYY-MM ~ YYYY-MM
            r'(\d{4}[-./]\d{1,2})\s*[-~–]\s*(현재|present|now)',   # YYYY-MM ~ 현재
            r'(\d{4})\s*[-~–]\s*(\d{4})',                          # YYYY ~ YYYY
            r'(\d{4})\s*[-~–]\s*(현재|present|now)'                # YYYY ~ 현재
        ]

    # --------------------- 헬퍼 메서드 리팩토링 ---------------------
    def _process_text(self, text: str, pattern: str, replacement: str = ' ') -> str:
        """텍스트 처리 공통 함수"""
        return re.sub(pattern, replacement, text).strip()

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 간소화"""
        text = self._process_text(text, r'\s+', ' ')
        return self._process_text(text, r'[^\w\s\-–—.,()/]', '')

    def _parse_date_component(self, date_str: str) -> Optional[pd.Timestamp]:
        """날짜 파싱 공통 로직"""
        try:
            return pd.to_datetime(date_str)
        except:
            return None

    # --------------------- 주요 기능 모듈화 ---------------------
    def _handle_date_ranges(self, text: str) -> Tuple[int, List[Tuple[datetime, datetime]]]:
        """날짜 범위 처리 핵심 로직"""
        date_ranges = self._extract_explicit_date_ranges(text)
        
        if not date_ranges:
            inferred_ranges = self._infer_date_ranges_from_text(text)
            date_ranges.extend(inferred_ranges)
            
        return self._calculate_final_duration(date_ranges, text)

    def _extract_explicit_date_ranges(self, text: str) -> List[Tuple[datetime, datetime]]:
        """명시적 날짜 범위 추출"""
        date_ranges = []
        for pattern in self.date_range_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = self._parse_match(match)
                if start and end:
                    date_ranges.append((start, end))
        return date_ranges

    def _parse_match(self, match) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """날짜 매치 파싱"""
        start_str = match.group(1)
        end_str = match.group(2)
        
        start_date = self._parse_date_component(start_str) if start_str else None
        end_date = (pd.Timestamp.now() if end_str in ['현재', 'present', 'now'] 
                   else self._parse_date_component(end_str))
        
        if start_date and end_date and start_date <= end_date:
            if end_date > pd.Timestamp.now():
                end_date = pd.Timestamp.now()
            return start_date, end_date
            
        return None, None

    def _infer_date_ranges_from_text(self, text: str) -> List[Tuple[datetime, datetime]]:
        """텍스트에서 날짜 범위 추론"""
        dates = self.extract_dates(text)
        date_ranges = []
        
        if len(dates) >= 2:
            # 정렬된 날짜에서 날짜 쌍 추론
            sorted_dates = [pd.to_datetime(d) for d in dates if d != 'present']
            sorted_dates = [d for d in sorted_dates if d <= pd.Timestamp.now()]
            
            if len(sorted_dates) >= 2:
                start = min(sorted_dates)
                end = (pd.Timestamp.now() if 'present' in dates 
                      else max(sorted_dates))
                
                # 기간이 합리적인 범위인지 확인 (1개월 ~ 5년)
                duration_days = (end - start).days
                if 30 <= duration_days <= 365 * 5:
                    date_ranges.append((start, end))
                    
        return date_ranges

    def _calculate_final_duration(self, date_ranges: List[Tuple[datetime, datetime]], text: str) -> Tuple[int, List[Tuple[datetime, datetime]]]:
        """최종 기간 계산"""
        if not date_ranges:
            return 0, []
            
        # 프로젝트 섹션 여부 확인
        is_project = ("project" in text.lower() and 
                     not any(keyword in text.lower() for keyword in ["work", "employment", "company", "career"]))
        
        # 날짜 범위 병합
        months, merged_ranges = self.merge_date_ranges(date_ranges)
        
        # 프로젝트 섹션은 가중치 적용
        if is_project:
            months = int(months * 0.25)  # 프로젝트는 25%만 경력으로 인정
        
        return months, merged_ranges

    def merge_date_ranges(self, date_ranges: List[Tuple[datetime, datetime]]) -> Tuple[int, List[Tuple[datetime, datetime]]]:
        """중복 기간 병합 알고리즘 - 개선된 정확도"""
        if not date_ranges:
            return 0, []
        
        # 날짜 범위 정렬
        sorted_ranges = sorted(date_ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # 중복 또는 인접 기간 병합 (한달 이내 간격은 연속으로 간주)
            if current_start <= last_end + pd.DateOffset(months=1):
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        # 총 개월 수 계산 - 정확한 월 단위 계산
        total_months = 0
        for start, end in merged:
            # 연도 차이에 12를 곱하고 월 차이를 더함 (정확한 월 계산)
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if end.day >= start.day:  # 날짜까지 고려
                months += 1
            total_months += max(0, months)
        
        return total_months, merged

    # --------------------- 시각화 로직 개선 ---------------------
    def _setup_timeline_style(self):
        """타임라인 스타일 설정 공통화"""
        plt.figure(figsize=(15, 8))
        plt.gca().invert_yaxis()
        plt.title('경력 및 프로젝트 타임라인', pad=20, fontsize=14)
        plt.gcf().autofmt_xdate()

    def _create_timeline_entry(self, exp: dict, y_pos: int):
        """타임라인 엔트리 생성기"""
        plt.barh(y_pos, exp['duration'], 
                left=exp['start'], height=0.4,
                color=self.colors[exp['type']], alpha=0.8)
        self._add_timeline_annotation(exp, y_pos)

    def _add_timeline_annotation(self, exp: dict, y_pos: int):
        """타임라인 주석 추가"""
        duration_text = f"({int(exp['duration'])}개월)"
        plt.text(exp['start'], y_pos,
                f" {exp['text'][:60]}... {duration_text}",
                verticalalignment='center',
                fontsize=9)

    # --------------------- 출력 로직 최적화 ---------------------
    def _print_tech_summary(self, tech_stack: dict):
        """기술 스택 요약 출력 공통화"""
        if not tech_stack:
            return
            
        print("\n💻 사용 기술:")
        for category, techs in tech_stack.items():
            if techs:
                print(f"  • {category}: {', '.join(sorted(techs))}")

    def _print_section_header(self, title: str):
        """섹션 헤더 출력 표준화"""
        print(f"\n📌 {title.upper()} 섹션 요약")
        print("-" * 50)

    # --------------------- 주요 프로세스 리팩토링 ---------------------
    def process_resume(self) -> Dict:
        """이력서 처리 프로세스 최적화"""
        documents = self.load_resume()
        if not documents:
            return {}

        nodes = self.node_parser.get_nodes_from_documents(documents)
        sections = self._process_nodes(nodes)
        self._calculate_total_experience(sections)
        
        return sections

    def _calculate_total_experience(self, sections: Dict) -> None:
        """전체 경력 기간 계산"""
        all_date_ranges = []
        
        # 경력 및 프로젝트 섹션에서 날짜 범위 수집
        for section_type in ["experience", "projects"]:
            for content in sections.get(section_type, []):
                text = content.get("text", "")
                _, date_ranges = self._handle_date_ranges(text)
                all_date_ranges.extend(date_ranges)
        
        # 전체 경력 기간 계산
        total_months, merged_ranges = self.merge_date_ranges(all_date_ranges)
        
        # 결과 저장
        sections['total_experience'] = [{
            "total_months": total_months,
            "merged_ranges": merged_ranges
        }]

    def _process_nodes(self, nodes) -> defaultdict:
        """노드 처리 로직 분리"""
        sections = defaultdict(list)
        for node in nodes:
            section_data = self._analyze_node(node)
            if section_data:
                sections[section_data['type']].append(section_data['content'])
        return sections

    def _analyze_node(self, node) -> Optional[Dict]:
        """노드 분석 및 데이터 추출"""
        text = node.text
        if not text.strip():
            return None
            
        section_type = self.classify_section(text)
        key_points = self.extract_key_points(text)
        dates = self.extract_dates(text)
        
        # 경력 기간 계산
        duration = 0
        if section_type in ["experience", "projects"]:
            duration, _ = self._handle_date_ranges(text)
        
        return {
            'type': section_type,
            'content': {
                "text": self.preprocess_text(text),
                "key_points": key_points,
                "dates": dates,
                "duration": duration
            }
        }

    def load_resume(self) -> List[Document]:
        """이력서 PDF 파일 로드 - RAG 향상을 위한 개선"""
        try:
            # 향상된 추출 옵션으로 PDF 로드
            reader = SimpleDirectoryReader(
                input_dir=self.pdf_directory,
                recursive=True,
                filename_as_id=True,
                required_exts=[".pdf"],
                file_metadata=lambda filename: {"source": filename}
            )
            
            documents = reader.load_data()
            
            if not documents:
                print("⚠️ 이력서를 불러오지 못했습니다.")
                return []
                
            print(f"✅ 이력서를 성공적으로 불러왔습니다. (총 {len(documents)}개 문서)")
            return documents
        except Exception as e:
            print(f"⚠️ 이력서 로드 중 오류 발생: {e}")
            return []
    
    def classify_section(self, text: str) -> str:
        """텍스트의 섹션 분류"""
        text = text.lower()
        max_score = 0
        best_section = "other"
        
        # 각 섹션별 키워드 매칭 점수 계산
        for section, keywords in self.section_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > max_score:
                max_score = score
                best_section = section
        
        return best_section
    
    def extract_key_points(self, text: str) -> List[str]:
        """텍스트에서 주요 포인트 추출"""
        points = []
        
        # 글머리 기호로 시작하는 라인 추출
        bullet_points = re.findall(r'[•▪-]\s*([^\n]+)', text)
        points.extend([p for p in bullet_points if isinstance(p, str)])
        
        # 날짜로 시작하는 라인 추출
        date_points = []
        for pattern in self.date_patterns:
            if isinstance(pattern, str) and pattern in ['현재', 'present', 'now']:
                continue
            matches = re.findall(f'{pattern}[^\\n]+', text)
            date_points.extend([m for m in matches if isinstance(m, str)])
        points.extend(date_points)
        
        # 중복 제거 및 전처리
        points = [self.preprocess_text(point) for point in points if isinstance(point, str)]
        points = [p for p in points if p and len(p) > 5]  # 짧은 텍스트 및 빈 문자열 제거
        points = [p for p in points if not p.isdigit()]  # 숫자로만 된 텍스트 제거
        points = list(set(points))  # 중복 제거
        
        return points
    
    def extract_tech_stack(self, text: str) -> Dict[str, Set[str]]:
        """기술 스택 추출 및 분류"""
        text = text.lower()
        tech_stack = defaultdict(set)
        
        for category, keywords in self.tech_categories.items():
            for tech in keywords:
                if tech in text:
                    tech_stack[category].add(tech)
        
        return tech_stack

    def visualize_tech_stack(self, all_tech_stack: Dict[str, Set[str]]) -> None:
        """기술 스택 시각화"""
        plt.figure(figsize=(12, 6))
        categories = list(all_tech_stack.keys())
        counts = [len(techs) for techs in all_tech_stack.values()]
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=counts, y=categories)
        plt.title('기술 스택 분포', fontsize=12)
        plt.xlabel('기술 수', fontsize=10)
        
        plt.subplot(1, 2, 2)
        all_techs = ' '.join([' '.join(techs) for techs in all_tech_stack.values()])
        if all_techs:
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf' if platform.system() == 'Darwin' else None
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                font_path=font_path,
                max_font_size=100
            ).generate(all_techs)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('기술 스택 워드클라우드', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('tech_stack_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def extract_company_name(self, text: str) -> str:
        """텍스트에서 회사명 추출"""
        # 회사명 패턴
        company_patterns = [
            r'([가-힣a-zA-Z0-9\s]+)(주식회사|㈜|Corp\.|Inc\.|Ltd\.|Company|에서)',
            r'(NC\s*SOFT|COM2US|넥슨|카카오|네이버|라인|쿠팡|배달의민족|토스)',
            r'([가-힣a-zA-Z0-9\s]+)(기업|회사|그룹|corporation|corp|inc|limited|ltd)',
            r'([A-Z][A-Za-z0-9\s]+)(Technologies|Software|Games|Entertainment|Solutions)',
            r'([가-힣a-zA-Z0-9\s]+)(연구소|연구원|센터|연구실|연구팀)',
            r'([가-힣a-zA-Z0-9\s]+)(대학교|학교|학원|교육원)',
        ]
        
        for pattern in company_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company = match.group(1).strip() if match.group(1) else match.group(0).strip()
                if company and len(company) >= 2:  # 최소 2글자 이상
                    # 불필요한 문자 제거
                    company = re.sub(r'[^\w\s]', '', company)
                    company = re.sub(r'\s+', ' ', company).strip()
                    return company
        return ""

    def visualize_experience_timeline(self, sections: Dict) -> None:
        """경력 타임라인 시각화"""
        if not sections.get("projects"):
            print("⚠️ 프로젝트 정보가 없습니다.")
            return
            
        experiences = []
        
        # 경력 정보 수집 - projects만 포함
        for section_type in ["projects"]:
            for exp in sections.get(section_type, []):
                text = exp["text"]
                dates = self.extract_dates(text)
                
                if len(dates) < 1:  # 날짜 정보가 없는 경우 건너뛰기
                    continue
                
                try:
                    # 날짜 파싱
                    dates = [d for d in dates if d != 'present']
                    if not dates:
                        continue
                        
                    start_date = pd.to_datetime(min(dates))
                    
                    # 종료일 처리
                    if 'present' in self.extract_dates(text):
                        end_date = pd.Timestamp.now()
                    else:
                        end_date = pd.to_datetime(max(dates))
                        
                    # 미래 날짜 처리
                    if end_date > pd.Timestamp.now() + pd.DateOffset(years=1):
                        end_date = pd.Timestamp.now()
                    
                    duration = (end_date - start_date).days / 30  # 개월 수로 변환
                    if duration <= 0:  # 잘못된 기간 제외
                        continue
                    
                    # 회사명 추출
                    company_name = self.extract_company_name(text)
                    if not company_name:
                        company_name = f"{section_type.capitalize()}"  # 회사명이 없으면 섹션 타입으로 대체
                    
                    # 제목 추출 (첫 번째 줄 또는 키포인트)
                    title = exp["key_points"][0] if exp["key_points"] else text.split('\n')[0]
                    title = re.sub(r'\s+', ' ', title).strip()  # 공백 정리
                    title = re.sub(r'^\d+[\s.)-]*', '', title)  # 숫자로 시작하는 부분 제거
                    
                    experiences.append({
                        'start': start_date,
                        'end': end_date,
                        'duration': int(duration),
                        'text': title,
                        'type': section_type,
                        'company': company_name
                    })
                    
                except Exception as e:
                    print(f"⚠️ 날짜 처리 중 오류 발생: {e}")
                    continue
        
        if not experiences:
            print("⚠️ 시각화할 경력 데이터가 없습니다.")
            return
        
        # 회사명과 시작일 기준으로 정렬
        experiences.sort(key=lambda x: (x['company'], x['start']))
        
        # 회사별로 그룹화
        companies = {}
        for exp in experiences:
            company = exp['company']
            if company not in companies:
                companies[company] = []
            companies[company].append(exp)
        
        # 타임라인 그래프 생성
        plt.figure(figsize=(15, max(8, len(companies) * 1.2)))  # 회사 수에 따라 그래프 높이 조정
        plt.title('회사별 프로젝트 타임라인', pad=20, fontsize=14)
        plt.gcf().autofmt_xdate()
        
        # 색상 및 스타일 설정
        self.colors = {'projects': '#3498db'}
        self.alpha = 0.8
        self.bar_height = 0.3
        
        # Y축 위치 계산 (회사별로 간격 두기)
        company_names = list(companies.keys())
        y_positions = {}
        
        # 모든 항목에 Y 위치 할당
        current_pos = 0
        for i, company in enumerate(company_names):
            exps = companies[company]
            for j, exp in enumerate(exps):
                y_positions[id(exp)] = current_pos
                current_pos += 1
            # 회사 사이에 간격 추가
            if i < len(company_names) - 1:
                current_pos += 0.5
        
        # 모든 항목 그리기
        for company in company_names:
            for exp in companies[company]:
                y_pos = y_positions[id(exp)]
                # 막대 그래프 그리기
                plt.barh(y_pos, width=(exp['end'] - exp['start']).days / 30, 
                         left=exp['start'], height=self.bar_height,
                         color=self.colors[exp['type']], alpha=self.alpha)
                
                # 텍스트 추가
                duration_text = f"({exp['duration']}개월)"
                plt.text(exp['start'], y_pos, 
                         f" {exp['text'][:40]}... {duration_text}",
                         verticalalignment='center', fontsize=8)
        
        # X축 설정 (년도와 월 표시)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Y축 레이블 설정 - 회사명으로 표시
        company_positions = []
        company_labels = []
        
        current_pos = 0
        for i, company in enumerate(company_names):
            exps = companies[company]
            # 회사의 중간 위치를 계산
            mid_pos = current_pos + len(exps) / 2 - 0.5
            company_positions.append(mid_pos)
            company_labels.append(company)
            current_pos += len(exps)
            if i < len(company_names) - 1:
                current_pos += 0.5
                
        plt.yticks(company_positions, company_labels, fontsize=10)
        
        # 그리드 및 범례
        plt.grid(True, alpha=0.3, axis='x')
        
        # 수동으로 범례 생성 - 프로젝트만 표시
        legend_elements = [
            Patch(facecolor=self.colors['projects'], label='프로젝트', alpha=self.alpha)
        ]
        plt.legend(handles=legend_elements, title='유형', loc='upper right')
        
        # 여백 조정
        plt.tight_layout()
        
        # 저장
        plt.savefig('experience_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 경력 타임라인 시각화가 생성되었습니다.")

    def print_section_summary(self, section_name: str, contents: List[Dict]) -> None:
        """섹션별 요약 정보 출력"""
        if not contents:
            return
            
        self._print_section_header(section_name)
        
        if section_name == 'total_experience':
            total_months = contents[0].get('total_months', 0)
            merged_ranges = contents[0].get('merged_ranges', [])
            
            print(f"📅 전체 경력 기간: {total_months}개월 ({total_months/12:.1f}년)")
            if merged_ranges:
                print("\n🔍 병합된 경력 기간:")
                for start, end in merged_ranges:
                    print(f"  • {start.strftime('%Y-%m')} ~ {end.strftime('%Y-%m')} ({int((end - start).days / 30)}개월)")
            return
        
        # 기간 정보
        dates = [date for content in contents for date in content.get("dates", [])]
        if dates:
            print(f"📅 전체 기간: {min(dates)} ~ {max(dates)}")
        
        # 주요 키워드
        all_points = [point for content in contents for point in content.get("key_points", [])]
        if all_points:
            print("\n🔑 주요 키워드:")
            for point in sorted(set(all_points))[:5]:  # 상위 5개만 출력
                print(f"  • {point}")
        
        # 기술 스택
        tech_stack = defaultdict(set)
        for content in contents:
            current_tech = self.extract_tech_stack(content.get("text", ""))
            for category, techs in current_tech.items():
                tech_stack[category].update(techs)
        
        self._print_tech_summary(tech_stack)
        
        print("-" * 50)

    def generate_summary(self) -> None:
        """이력서 요약 생성 및 출력"""
        sections = self.process_resume()
        if not sections:
            return
        
        print("\n📑 이력서 분석 결과")
        print("=" * 50)
        
        # 전체 통계
        total_experience = sections.get('total_experience', [{}])[0].get('total_months', 0)
        total_projects = len(sections.get("projects", []))
        total_publications = len(sections.get("publications", []))
        
        print(f"\n📊 전체 통계")
        print("-" * 50)
        print(f"• 총 경력: {total_experience}개월 ({total_experience/12:.1f}년)")
        print(f"• 프로젝트: {total_projects}개")
        print(f"• 논문/발표: {total_publications}개")
        
        # 각 섹션별 상세 요약
        for section_name, contents in sections.items():
            self.print_section_summary(section_name, contents)
        
        # 전체 기술 스택 수집 및 분석
        all_tech_stack = defaultdict(set)
        for content in sections.get("skills", []) + sections.get("experience", []) + sections.get("projects", []):
            tech_stack = self.extract_tech_stack(content["text"])
            for category, techs in tech_stack.items():
                all_tech_stack[category].update(techs)
        
        # 기술 스택 총합 출력
        if all_tech_stack:
            print("\n🔧 전체 기술 스택 분석")
            print("-" * 50)
            total_techs = sum(len(techs) for techs in all_tech_stack.values())
            print(f"총 보유 기술: {total_techs}개")
            
            for category, techs in all_tech_stack.items():
                if techs:
                    print(f"\n• {category} ({len(techs)}개)")
                    print(f"  - {', '.join(sorted(techs))}")
        
        # 시각화 생성
        self.visualize_tech_stack(all_tech_stack)
        self.visualize_experience_timeline(sections)
        
        # 결과 파일 저장
        self.save_to_csv(sections)
        
        print("\n📁 생성된 파일 목록")
        print("-" * 50)
        print("1. resume_summary.csv - 상세 분석 데이터")
        print("2. tech_stack_analysis.png - 기술 스택 시각화")
        print("3. experience_timeline.png - 경력 타임라인")

    def save_to_csv(self, sections: Dict) -> None:
        """분석 결과를 CSV 파일로 저장"""
        df_data = []
        for section_name, contents in sections.items():
            if section_name == 'total_experience':
                continue
                
            for content in contents:
                tech_stack = self.extract_tech_stack(content.get("text", ""))
                df_data.append({
                    "섹션": section_name,
                    "날짜": ", ".join(content.get("dates", [])),
                    "기간(개월)": content.get("duration", 0),
                    "주요내용": "\n".join(content.get("key_points", [])),
                    "기술스택": str(dict(tech_stack)),
                    "전체내용": content.get("text", "")
                })
        
        df = pd.DataFrame(df_data)
        df.to_csv("resume_summary.csv", index=False, encoding='utf-8-sig')
        print(f"\n✅ 분석 결과가 CSV 파일로 저장되었습니다. (총 {len(df_data)}개 항목)")

    def normalize_date(self, date_str: str) -> Optional[str]:
        """날짜 문자열 정규화"""
        try:
            # 현재 날짜 처리
            if date_str.lower() in ['현재', 'present', 'now']:
                return pd.Timestamp.now().strftime('%Y-%m')
            
            # 월 이름을 숫자로 변환
            date_str = date_str.lower()
            for month, num in [('jan', '01'), ('feb', '02'), ('mar', '03'), ('apr', '04'),
                             ('may', '05'), ('jun', '06'), ('jul', '07'), ('aug', '08'),
                             ('sep', '09'), ('oct', '10'), ('nov', '11'), ('dec', '12')]:
                date_str = date_str.replace(month, num)
            
            # 다양한 구분자 통일
            date_str = re.sub(r'[./]', '-', date_str)
            
            # 한글 제거
            date_str = re.sub(r'[년월일]', '-', date_str)
            date_str = re.sub(r'-+', '-', date_str)
            date_str = date_str.strip('-')
            
            # MM-YYYY를 YYYY-MM으로 변환
            if re.match(r'^(0?[1-9]|1[0-2])-\d{4}$', date_str):
                month, year = date_str.split('-')
                date_str = f"{year}-{int(month):02d}"
            
            # 년도만 있는 경우 처리
            if re.match(r'^\d{4}$', date_str):
                date_str += '-01'
            
            # 날짜 파싱 및 포맷팅
            date = pd.to_datetime(date_str)
            
            # 미래 날짜 처리 (현재 기준 1년 이후는 제외)
            if date > pd.Timestamp.now() + pd.DateOffset(years=1):
                return None
                
            return date.strftime('%Y-%m')
        except:
            return None

    def extract_dates(self, text: str) -> List[str]:
        """텍스트에서 날짜 추출"""
        dates = []
        text = text.lower()
        
        # 현재 날짜 처리
        if any(keyword in text for keyword in ['현재', 'present', 'now']):
            dates.append('present')
        
        # 다른 날짜 패턴 처리
        for pattern in self.date_patterns:
            if pattern in ['현재', 'present', 'now']:
                continue
                
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group()
                normalized_date = self.normalize_date(date_str)
                if normalized_date:
                    dates.append(normalized_date)
        
        return sorted(list(set(dates)))

def main():
    print("\n🔍 이력서 분석을 시작합니다...")
    ResumeSummarizer().generate_summary()
    print("\n✨ 이력서 분석이 완료되었습니다.")

if __name__ == "__main__":
    main() 