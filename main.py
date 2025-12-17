import json
import re
from typing import List, Optional, Dict, Tuple, Set, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel

# ============================================================
#   МОДЕЛИ ДАННЫХ
# ============================================================

class LawLink(BaseModel):
    law_id: int
    article: str
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None

    class Config:
        frozen = True

    def __eq__(self, other):
        if not isinstance(other, LawLink):
            return False
        def norm(val): return str(val).strip().lower() if val else ""
        return (
            self.law_id == other.law_id and
            norm(self.article) == norm(other.article) and
            norm(self.point_article) == norm(other.point_article) and
            norm(self.subpoint_article) == norm(other.subpoint_article)
        )
    
    def __hash__(self):
        def norm(val): return str(val).strip().lower() if val else ""
        return hash((
            self.law_id, 
            norm(self.article), 
            norm(self.point_article), 
            norm(self.subpoint_article)
        ))

# Модели для API (остались прежними, но ссылаются на новый LawLink)
class LinksResponse(BaseModel):
    links: List[LawLink]

class TextRequest(BaseModel):
    text: str

# ============================================================
#   2. PROCESSOR (ВАШ НОВЫЙ КОД)
# ============================================================

class TextProcessor:
    def __init__(self, law_aliases: Dict[str, List[str]] = None):
        # JSON загружает ключи как строки, поэтому аннотация Dict[str, ...]
        self.law_aliases = law_aliases or {}
        
        self.conjunctions = {'и', 'а', 'или', 'либо', 'также'}
        self.stop_markers = {
            'в', 'на', 'к', 'от', 'до', 'при', 'из', 'со', 'об', 'по', 'за', 'под', 
            'ст.', 'п.', 'ч.', 'пп.', 'статья', 'пункт', 'часть', 'подпункт'
        }
        
        self.regexes = self._compile_structural_regex()
        self.law_patterns = self._compile_law_patterns()

    def _compile_structural_regex(self) -> Dict[str, re.Pattern]:
        flags = re.IGNORECASE | re.UNICODE
        return {
            'subpoint': re.compile(r'(?<!\w)(?:подпункт[а-я]*|подпп?\.?|пп\.?)(?!\w)\s*', flags),
            'point': re.compile(r'(?<!\w)(?:пункт[а-я]*|част[а-я]+|ч\.|п\.?)(?!\w|п)\s*', flags),
            'article': re.compile(r'(?<!\w)(?:стать[а-я]+|ст\.?)(?!\w)\s*', flags),
            'value': re.compile(
                r'(?<!\w)(?!\d{2}\.\d{2}\.\d{4})(?:[\w\.\-]*\d[\w\.\-]*|[а-яёa-z])(?!\w)', 
                flags
            ),
            'date': re.compile(r'\d{2}\.\d{2}\.\d{4}', flags)
        }

    def _compile_law_patterns(self) -> List[Tuple[re.Pattern, int]]:
        raw_aliases = []
        for law_id, aliases_list in self.law_aliases.items():
            for alias in aliases_list:
                clean = re.sub(r'(№|N|No)\s+(\d)', r'\1\2', alias.strip(), flags=re.IGNORECASE)
                clean = re.sub(r'\s+', ' ', clean)
                # law_id из JSON приходит строкой, конвертируем в int
                raw_aliases.append((clean, int(law_id)))
        
        raw_aliases.sort(key=lambda x: len(x[0]), reverse=True)

        patterns = []
        rf_pattern_str = r'(?:РФ|Росси[а-я]+(?:\s+Федераци[а-я]+)?)'

        for alias, law_id in raw_aliases:
            parts = alias.split()
            regex_parts = []
            
            for part in parts:
                clean_part = part.strip('.,;').upper()
                
                if any(q in part for q in ['"', '«', '»', '“', '”']):
                    escaped = re.escape(part)
                    flexible = re.sub(r'\\?[\"«»“”]', r'[\"«»“”]', escaped)
                    regex_parts.append(flexible)
                    continue

                if not clean_part:
                    continue

                if clean_part in ["РФ", "РОССИИ", "РОССИЙСКОЙ", "ФЕДЕРАЦИИ", "ФЕДЕРАЦИЯ"]:
                    if regex_parts and regex_parts[-1] == rf_pattern_str:
                        continue
                    regex_parts.append(rf_pattern_str)
                elif re.match(r'^\d{2}\.\d{2}\.\d{4}$', clean_part):
                    regex_parts.append(re.escape(clean_part))
                elif re.match(r'^(?:N|№|НОМЕР).*', clean_part):
                    if len(clean_part) > 1 and (clean_part.startswith('N') or clean_part.startswith('№')):
                        num_part = clean_part[1:]
                        regex_parts.append(r'(?:№|N|номер)\s*' + re.escape(num_part))
                    else:
                        regex_parts.append(r'(?:№|N|номер)\s*')
                elif re.match(r'^[\d\-\.]+$', clean_part):
                    regex_parts.append(re.escape(clean_part))
                else:
                    if len(clean_part) <= 3:
                        token = re.escape(part)
                    else:
                        root_len = max(3, len(clean_part) - 2)
                        root = clean_part[:root_len]
                        token = re.escape(root) + r'[а-яё]*'
                    regex_parts.append(token)
            
            if not regex_parts:
                continue

            pattern_str = r'(?<!\w)' + r'\s+'.join(regex_parts)
            p_conn = r'[\s,]+'
            p_num = r'(?:№|N|номер)\s*[\d\-\w]+'
            p_date = r'(?:от|дата|принят)\s+[\d\.]+(?:\s+[а-я]+)?(?:\s+\d{4})?'
            p_title = r'(?:«[^»]+»|"[^"]+")' 
            p_tail_rf = r'(?:\s+' + rf_pattern_str + r')?'

            full_pattern = (
                pattern_str + 
                p_tail_rf + 
                f'(?:{p_conn}(?:{p_num}|{p_date}|{p_title}))*' + 
                r'(?!\w)'
            )
            patterns.append((re.compile(full_pattern, re.IGNORECASE | re.DOTALL), law_id))
            
        return patterns

    def find_law_positions(self, text: str) -> List[Tuple[int, int, int]]:
        found = []
        used_ranges = set()

        for pattern, law_id in self.law_patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                is_overlap = any(
                    (r_start <= start < r_end) or (r_start < end <= r_end) or (start <= r_start and end >= r_end)
                    for r_start, r_end in used_ranges
                )
                if not is_overlap:
                    found.append((start, end, law_id))
                    used_ranges.add((start, end))
        
        return sorted(found, key=lambda x: x[0])

    def parse_values_smart(self, text_segment: str, is_subpoint_context: bool = False) -> List[str]:
        if not text_segment:
            return []
            
        clean_values = []
        raw_tokens = re.split(r'(\s+|[,;])', text_segment)
        tokens = [t.strip() for t in raw_tokens if t.strip()]
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            token_lower = token.lower()

            if token in {',', ';'}:
                i += 1
                continue

            if self.regexes['date'].fullmatch(token):
                break

            val_candidate = token.rstrip('.')
            val_match = self.regexes['value'].fullmatch(val_candidate)
            
            if val_match:
                is_conjunction = False
                if token_lower in self.conjunctions:
                    is_conjunction = True
                    if is_subpoint_context:
                        next_token = tokens[i+1] if i + 1 < len(tokens) else ""
                        is_end_or_marker = (not next_token or any(next_token.lower().startswith(m) for m in self.stop_markers))
                        next_is_value = bool(self.regexes['value'].fullmatch(next_token.rstrip('.')))

                        if is_end_or_marker and not next_is_value:
                            is_conjunction = False
                
                if is_conjunction:
                    i += 1
                    continue
                
                clean_values.append(val_candidate)
                i += 1
                continue
            
            if token_lower not in self.conjunctions:
                break
            i += 1
            
        return clean_values

    def _extract_hierarchy(self, text_segment: str) -> List[Dict]:
        results = []
        art_matches = list(self.regexes['article'].finditer(text_segment))
        if not art_matches:
            return []

        class DummyMatch:
            def start(self): return len(text_segment)
            def end(self): return len(text_segment)

        all_boundaries = art_matches + [DummyMatch()]
        prev_boundary_end = 0

        for i, art_match in enumerate(art_matches):
            next_art_start = all_boundaries[i+1].start()
            
            values_segment = text_segment[art_match.end():next_art_start]
            articles_list = self.parse_values_smart(values_segment, is_subpoint_context=False)
            
            context_segment = text_segment[prev_boundary_end:art_match.start()]
            points_structure = self._extract_points(context_segment)

            for art_val in articles_list:
                results.append({
                    'article': art_val,
                    'points': points_structure
                })
            
            prev_boundary_end = art_match.end()
        return results

    def _extract_points(self, text_segment: str) -> List[Dict]:
        point_matches = list(self.regexes['point'].finditer(text_segment))
        if not point_matches:
            subpoints = self._extract_subpoints(text_segment)
            if subpoints:
                return [{'point': None, 'subpoints': subpoints}]
            return []

        results = []
        class DummyMatch:
            def start(self): return len(text_segment)
        
        all_boundaries = point_matches + [DummyMatch()]
        prev_boundary_end = 0

        for i, pt_match in enumerate(point_matches):
            next_pt_start = all_boundaries[i+1].start()
            values_segment = text_segment[pt_match.end():next_pt_start]
            points_list = self.parse_values_smart(values_segment, is_subpoint_context=False)
            context_segment = text_segment[prev_boundary_end:pt_match.start()]
            subpoints_list = self._extract_subpoints(context_segment)

            for pt_val in points_list:
                results.append({
                    'point': pt_val,
                    'subpoints': subpoints_list
                })
            prev_boundary_end = pt_match.end()
        return results

    def _extract_subpoints(self, text_segment: str) -> List[str]:
        sub_matches = list(self.regexes['subpoint'].finditer(text_segment))
        if not sub_matches:
            return []
        last_match = sub_matches[-1]
        values_segment = text_segment[last_match.end():]
        return self.parse_values_smart(values_segment, is_subpoint_context=True)

    def extract_law_references(self, text: str) -> Set[LawLink]:
        law_positions = self.find_law_positions(text)
        links = set()
        
        prev_law_end = 0
        last_law_id = None

        for law_start, law_end, law_id in law_positions:
            segment = text[prev_law_end:law_start]
            if segment.strip():
                hierarchy = self._extract_hierarchy(segment)
                for item in hierarchy:
                    self._add_links(links, item, law_id)
            prev_law_end = law_end
            last_law_id = law_id

        tail_segment = text[prev_law_end:]
        if tail_segment.strip() and last_law_id:
            hierarchy = self._extract_hierarchy(tail_segment)
            for item in hierarchy:
                self._add_links(links, item, last_law_id)

        return links

    def _add_links(self, links_set: Set[LawLink], item: Dict, law_id: int):
        art = item['article']
        points_data = item['points']
        if not points_data:
            links_set.add(LawLink(law_id=law_id, article=art))
        else:
            for p_data in points_data:
                pt = p_data['point']
                subs = p_data['subpoints']
                if not subs:
                    links_set.add(LawLink(law_id=law_id, article=art, point_article=pt))
                else:
                    for sub in subs:
                        links_set.add(LawLink(
                            law_id=law_id, 
                            article=art, 
                            point_article=pt, 
                            subpoint_article=sub
                        ))


# ============================================================
#   FASTAPI СЕРВИС
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    with open("law_aliases.json", "r", encoding="utf-8") as file:
        codex_aliases = json.load(file)

    # создаем процессор один раз при старте
    app.state.processor = TextProcessor(codex_aliases)
    print("🚀 Сервис запущен и готов к обработке текста...")
    yield
    # --- Shutdown ---
    del codex_aliases
    del app.state.processor
    print("🛑 Сервис завершает работу...")


def get_processor(request: Request) -> TextProcessor:
    return request.app.state.processor


app = FastAPI(
    title="Law Links Service",
    description="Сервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/detect", response_model=LinksResponse)
async def get_law_links(
    data: TextRequest,
    processor: TextProcessor = Depends(get_processor)
):
    """
    Принимает текст и возвращает список юридических ссылок
    """
    links = processor.extract_law_references(data.text)
    return LinksResponse(links=links)


@app.get("/health")
async def health_check():
    """
    Проверка состояния сервиса
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8978)
