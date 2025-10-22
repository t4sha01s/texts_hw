import json
import re
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel


import inspect

if not hasattr(inspect, "getargspec"):
    from collections import namedtuple
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def getargspec(func):
        from inspect import signature, Parameter
        params = signature(func).parameters
        args = [p for p in params if params[p].kind == Parameter.POSITIONAL_OR_KEYWORD]
        varargs = next((p for p in params if params[p].kind == Parameter.VAR_POSITIONAL), None)
        varkw = next((p for p in params if params[p].kind == Parameter.VAR_KEYWORD), None)
        defaults = tuple(p.default for p in params.values() if p.default is not Parameter.empty)
        return ArgSpec(args, varargs, varkw, defaults)
    inspect.getargspec = getargspec
    
import pymorphy2
from pymorphy2 import MorphAnalyzer

# ============================================================
#   МОДЕЛИ ДАННЫХ
# ============================================================

class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None


class LinksResponse(BaseModel):
    links: List[LawLink]


class TextRequest(BaseModel):
    text: str


# ============================================================
#   ОСНОВНОЙ КЛАСС ДЛЯ ОБРАБОТКИ ТЕКСТА
# ============================================================

class TextProcessor:
    def __init__(self, law_aliases: Dict = None):
        self.law_aliases = law_aliases or {}
        self.patterns = self.compile_patterns()
        self.morph = pymorphy2.MorphAnalyzer()

        # --- обратный индекс алиасов ---
        self.alias_to_id = {}
        for law_id, aliases in self.law_aliases.items():
            for alias in aliases:
                normalized_alias = self.normalize_law_name(alias)
                self.alias_to_id[normalized_alias] = int(law_id)

        # --- сортировка алиасов по длине ---
        self.sorted_aliases = sorted(self.alias_to_id.keys(), key=len, reverse=True)

    def compile_patterns(self) -> Dict[str, re.Pattern]:
        return {
            'article': re.compile(
                r'(?:ст|статья|статьи|статьей|статью|статье|ст\.)\s*(\d+(?:\.\d+)*)',
                re.IGNORECASE
            ),
            'point': re.compile(
                r'(?<!\w)(?:п|пункт|п\.)\s*(\d+|[а-я])(?!\w)',
                re.IGNORECASE
            ),
            'subpoint': re.compile(
                r'(?<!\w)(?:подп|подпункт|подп\.)\s*(\d+|[а-я])(?!\w)',
                re.IGNORECASE
            ),
        }

    def normalize_law_name(self, text: str) -> str:
        words = re.findall(r'\b[а-я]+\b', text.lower())
        normalized_words = []
        for word in words:
            try:
                parsed = self.morph.parse(word)[0]
                normalized_words.append(parsed.normal_form)
            except Exception:
                normalized_words.append(word)
        return ' '.join(normalized_words)

    def find_law_mentions(self, text: str) -> List[int]:
        normalized_text = self.normalize_law_name(text)
        found_ids = set()
        for alias in self.sorted_aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, normalized_text):
                found_ids.add(self.alias_to_id[alias])
        return list(found_ids)

    def extract_law_references(self, text: str) -> List[LawLink]:
        law_links = []
        law_ids = self.find_law_mentions(text)
        articles = self.patterns['article'].findall(text)
        points = self.patterns['point'].findall(text)
        subpoints = self.patterns['subpoint'].findall(text)

        for law_id in law_ids:
            for i, article in enumerate(articles):
                link = LawLink(law_id=law_id, article=article)
                if points and i < len(points):
                    link.point_article = points[i]
                if subpoints and i < len(subpoints):
                    link.subpoint_article = subpoints[i]
                law_links.append(link)

        # удаляем дубликаты
        unique_links = []
        seen = set()
        for link in law_links:
            key = (link.law_id, link.article, link.point_article, link.subpoint_article)
            if key not in seen:
                seen.add(key)
                unique_links.append(link)
        return unique_links


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
