import contextlib
import io
import os
from typing import List

from langchain_core.embeddings import Embeddings


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


class LocalBGEEmbeddings(Embeddings):
    """把本地 BGE 模型封装成 LangChain 可用的 Embeddings。"""

    def __init__(self, model_path: str, use_fp16: bool = False):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.model = None

    def _load_model(self):
        """延迟加载模型，避免 import 时就卡住。"""
        if self.model is not None:
            return self.model

        print(f"正在加载本地向量模型：{self.model_path}")
        from FlagEmbedding import FlagModel

        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
        except Exception:
            pass

        with contextlib.redirect_stderr(io.StringIO()):
            self.model = FlagModel(
                self.model_path,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=self.use_fp16,
            )

        print("本地向量模型加载完成")
        return self.model

    def _encode(self, texts: List[str]) -> List[List[float]]:
        model = self._load_model()
        result = model.encode(texts)

        # FlagModel 返回 ndarray，BGEM3FlagModel 返回 dict。
        if isinstance(result, dict):
            vectors = result["dense_vecs"]
        else:
            vectors = result

        return [vec.tolist() for vec in vectors]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print(f"正在向量化文档块，数量：{len(texts)}")
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]
