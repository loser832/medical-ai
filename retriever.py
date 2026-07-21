import json
from typing import Optional, Dict, List, Any
import torch
from config import (
    DEFAULT_FAISS_VERSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_VERSIONS,
    HF_LOCAL_FILES_ONLY,
    MODEL_NAME,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVER_GENERATION_CONFIG,
    RETRIEVER_MAIN_TOPK,
    RETRIEVER_MIN_SCORE,
    RETRIEVER_SUB_TOPK,
    VECTOR_RETRIEVER_TOP_K,
)
from utils import apply_thinking_instruction, strip_thinking_content
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains.llm import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
import regex as re
from transformers.generation.utils import GenerationConfig

_JSON_BLOCK_RE = re.compile(r"(\{(?:[^{}]|(?1))*\}|\[(?:[^\[\]]|(?0))*\])", re.S)

def _extract_first_json_block(text: str) -> str:
    """
    从文本中提取第一个完整 JSON 块（对象或数组）。
    """
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError("未在模型回复中找到 JSON。")
    return m.group(0)

def _safe_json_loads(text: str) -> Any:
    """
    安全解析 JSON（去除可能的 Markdown 包裹，并做一次常见问题修正）。
    """
    # 去掉 Markdown 代码围栏
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text.strip(), flags=re.I).strip()
        text = re.sub(r"```$", "", text.strip())
    # 直接尝试解析
    return json.loads(text)

class BgeReranker(BaseDocumentCompressor):
    model_name: str = Field(RERANK_MODEL, description="模型名称")
    top_n: int = Field(RERANK_TOP_N, description="保留的最大文档数")
    min_score: float = Field(RETRIEVER_MIN_SCORE, description="保留文档的最低阈值")
    device: str = Field("cuda", description="使用的设备 CPU/CUDA") 

    class Config:
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data) 
        if torch.cuda.is_available():
            self.device = "cuda"
        
        # 初始化模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=HF_LOCAL_FILES_ONLY,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                local_files_only=HF_LOCAL_FILES_ONLY,
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"模型加载失败: {str(e)}")
    
    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        **kwargs: Any,
    ) -> List[Document]:

        if len(documents) == 0:
            return []
        # 构造query-doc pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 批量编码
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**inputs).logits.view(-1)
            scores = torch.sigmoid(logits).cpu().numpy()

        # 组合文档与分数
        scored_docs = [(score, doc) for score, doc in zip(scores, documents)]
        
        filtered = [(s, d) for s, d in scored_docs if s >= self.min_score]
        sorted_docs = sorted(filtered, key=lambda x: x[0], reverse=True)
        
        # 保留top_n文档并更新元数据
        for score, doc in sorted_docs[:self.top_n]:
            doc.metadata["relevance_score"] = float(score)
            
        return [doc for _, doc in sorted_docs[:self.top_n]]
    
    

def decompose_question(question: str, model) -> List[Dict[str, Any]]:
    prompt = f"""
你是医学问答专家。请对下面的复杂问题进行“先思考、后输出”的分解。
目标：判断为了高质量回答与检索，**应该拆成几个问题**，并据此给出子问题列表。
如果拆解后的子问题超过5个，请你保留你认为最重要的5个子问题。

分解规则：
1) 先在推理如何拆分问题（请你再内心思考，不要给出推理过程），再输出JSON。
2) 子问题必须“可独立检索、语义清晰、互补不重复”，覆盖原问题的关键维度（病种/症状、鉴别点、检查/指标、干预/药物、适应证与禁忌、结局/随访、指南时效等）。
3) 对每个子问题给出：
   - sub_question: 子问题（保持全面、简洁、明确）       
   - why_needed: 说明该子问题存在的必要性

只输出一个 JSON 数组，勿添加其他说明。

原始问题：{question}
""".strip()
    messages = []
    messages.append({"role": "user", "content": apply_thinking_instruction(prompt)})
    config = RETRIEVER_GENERATION_CONFIG.copy()
    config.update({
        "messages": messages,
        "model": MODEL_NAME,
    })
    response = model.chat.completions.create(**config).choices[0].message.content
    response = strip_thinking_content(response)
    try:
        json_block = _extract_first_json_block(response)
        result = _safe_json_loads(json_block)
        assert isinstance(result, list), "期望得到一个 JSON 数组。"
        # 轻微校验与清洗
        cleaned = []
        for item in result:
            if not isinstance(item, dict): 
                continue
            sq = (item.get("sub_question") or "").strip()
            if not sq:
                continue
            cleaned.append({
                "sub_question": sq,
                "why_needed": (item.get("why_needed") or "").strip(),
            })
        return cleaned
    except Exception as e:
        print("分解解析错误:", e)
        return []

def decompose_question_dim(question: str, model,description):
    print(description)
    prompt = f"""您是{description}，为解决一个复杂医疗问题，你的工作是从你的专业角度出发，获取一些你的领域所需要的额外知识用于解决问题，不要考虑其他领域，。请对下面的复杂问题进行“先思考、后输出”的分解。
目标：从你的领域角度出发，判断为了高质量回答与检索，**应该拆成几个问题**，并据此给出子问题列表。
如果拆解后的子问题超过5个，请你保留你认为最重要的5个子问题。

分解规则：
1) 先在推理如何拆分问题（请你再内心思考，不要给出推理过程），再输出JSON。
2) 子问题必须“可独立检索、语义清晰、互补不重复”，覆盖原问题的关键维度（病种/症状、鉴别点、检查/指标、干预/药物、适应证与禁忌、结局/随访、指南时效等）。
3) 对每个子问题给出：
   - sub_question: 子问题（保持全面、简洁、明确）       
   - why_needed: 说明该子问题存在的必要性

只输出一个 JSON 数组，勿添加其他说明。

原始问题：{question}
""".strip()
    messages = []
    messages.append({"role": "user", "content": apply_thinking_instruction(prompt)})
    config = RETRIEVER_GENERATION_CONFIG.copy()
    config.update({
        "messages": messages,
        "model": MODEL_NAME,
    })
    response = model.chat.completions.create(**config).choices[0].message.content
    response = strip_thinking_content(response)
    try:
        json_block = _extract_first_json_block(response)
        result = _safe_json_loads(json_block)
        assert isinstance(result, list), "期望得到一个 JSON 数组。"
        # 轻微校验与清洗
        cleaned = []
        for item in result:
            if not isinstance(item, dict): 
                continue
            sq = (item.get("sub_question") or "").strip()
            if not sq:
                continue
            cleaned.append({
                "sub_question": sq,
                "why_needed": (item.get("why_needed") or "").strip(),
            })
        return cleaned
    except Exception as e:
        print("分解解析错误:", e)
        return []

def judge_and_polish(subquestions: List[str], model) -> List[Dict[str, Any]]:
    """
    输入: 子问题列表 (List[str])
    输出: 每个子问题附加 needs_polish, refined, polish_reason
    """
    prompt = f"""
你是医学问答专家。下面给出若干子问题，请你逐一判断是否需要润色：
- 如果子问题过于宽泛、缺少人群/疾病/时间/干预限定，或者存在术语歧义，请标记为需要润色。
- 如果需要润色，请对其进行更明确、更完整的改写（加入背景限定，但不要凭空编造）。
- 同时说明润色理由。
- 如果不需要润色，则保持原句即可。

输出严格为 JSON 数组，数组长度与输入一致。每个元素是一个对象，字段包括：
- sub_question: 原始子问题
- needs_polish: true/false
- refined_question: 润色后的子问题（若不需要润色则与原句一致）
- polish_reason: 润色的理由（若不需要则为空字符串）

输入子问题：
{json.dumps(subquestions, ensure_ascii=False, indent=2)}
    """.strip()

    messages = []
    messages.append({"role": "user", "content": apply_thinking_instruction(prompt)})
    config = RETRIEVER_GENERATION_CONFIG.copy()
    config.update({
        "messages": messages,
        "model": MODEL_NAME,
    })
    response = model.chat.completions.create(**config).choices[0].message.content
    response = strip_thinking_content(response)

    # 尝试提取 JSON 部分
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*", "", text).strip()
            text = text.removesuffix("```").strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            return text[start:end+1]
        return text

    try:
        json_block = _extract_json(response)
        result = json.loads(json_block)
        return result
    except Exception as e:
        print("解析出错:", e)
        return []


def rewrite_question_variants(
    question: str,
    model,
    rewrite_count: int = 3,
) -> List[str]:
    """为一个已分配的子问题生成固定数量的检索改写。"""
    if rewrite_count <= 0:
        return []

    prompt = f"""
你是医学知识库检索查询改写专家。请在不改变原意、不补充患者未知事实的前提下，
把下面的医学子问题改写为 {rewrite_count} 个彼此互补的检索查询。

要求：
1. 每个改写都必须能独立用于医学向量库检索；
2. 保留原问题中的人群、疾病、时间、检查、药物和干预限制；
3. 可以分别突出指南证据、诊断鉴别、治疗条件或风险，但不得虚构条件；
4. 恰好输出 {rewrite_count} 个不同的改写；
5. 只输出 JSON，格式为：{{"rewrites": ["改写1", "改写2", "改写3"]}}。

子问题：{question}
""".strip()

    candidates: List[str] = []
    try:
        config = RETRIEVER_GENERATION_CONFIG.copy()
        config.update({
            "messages": [{"role": "user", "content": apply_thinking_instruction(prompt)}],
            "model": MODEL_NAME,
        })
        response = model.chat.completions.create(**config).choices[0].message.content
        response = strip_thinking_content(response)
        json_block = _extract_first_json_block(response)
        result = _safe_json_loads(json_block)
        if isinstance(result, dict):
            raw_rewrites = result.get("rewrites", [])
        elif isinstance(result, list):
            raw_rewrites = result
        else:
            raw_rewrites = []

        for item in raw_rewrites:
            if isinstance(item, dict):
                item = item.get("rewrite") or item.get("question") or ""
            if isinstance(item, str) and item.strip():
                candidates.append(item.strip())
    except Exception as e:
        print("子问题改写解析错误:", e)

    # 去重，并在模型输出不足时使用不添加患者事实的通用检索表达补齐。
    rewrites: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            rewrites.append(candidate.strip())
        if len(rewrites) == rewrite_count:
            return rewrites

    fallback_rewrites = [
        f"关于以下医学问题的权威指南、适用条件和限制：{question}",
        f"针对以下医学问题的诊断鉴别、检查指标和临床证据：{question}",
        f"针对以下医学问题的干预选择、风险和随访证据：{question}",
    ]
    fallback_index = 1
    while len(rewrites) < rewrite_count:
        if fallback_index <= len(fallback_rewrites):
            candidate = fallback_rewrites[fallback_index - 1]
        else:
            candidate = f"医学知识库检索表达 {fallback_index}：{question}"
        fallback_index += 1
        normalized = candidate.lower()
        if normalized not in seen:
            seen.add(normalized)
            rewrites.append(candidate)

    return rewrites[:rewrite_count]

class Retriever:
    def __init__(
        self,
        base_version=DEFAULT_FAISS_VERSION,
        main_topk=RETRIEVER_MAIN_TOPK,
        sub_topk=RETRIEVER_SUB_TOPK,
        min_score=RETRIEVER_MIN_SCORE,
    ):
        self.main_topk = main_topk
        self.sub_topk = sub_topk
        self.min_score =min_score
        faiss_index_path = FAISS_INDEX_VERSIONS[base_version]
        self.guideline_bge_reranker = BgeReranker(model_name=RERANK_MODEL, top_n=RERANK_TOP_N, min_score=self.min_score)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.main_vector_db = FAISS.load_local(faiss_index_path, self.embeddings,allow_dangerous_deserialization=True )

    def retrieve_docs_for_question(self, question, topk):
        top_guideline_docs = []
        main_retriever = self.main_vector_db.as_retriever(search_kwargs={"k": VECTOR_RETRIEVER_TOP_K})
        main_compressor = self.guideline_bge_reranker
        compression_main_retriever = ContextualCompressionRetriever(
            base_compressor=main_compressor,
            base_retriever=main_retriever
        )
        main_docs = compression_main_retriever.invoke(question)
        top_guideline_docs = sorted(
            main_docs,
            key=lambda x: x.metadata['relevance_score'],
            reverse=True
        )
        def get_doc_content(docs, idx):
            try:
                return {
                    "q": docs[idx].metadata.get("qa_question", ""),
                    "a": docs[idx].page_content
                }
            except:
                return None
        if len(top_guideline_docs) == 0:
            return {
                "main_docs": None
            }
        else:
            return {
                "main_docs": [get_doc_content(top_guideline_docs, idx) for idx in range(min(topk, len(top_guideline_docs)))]
            }

    def retrieve_docs_for_subquestion(
        self,
        sub_question: str,
        model,
        rewrite_count: int = 3,
    ) -> Dict[str, Any]:
        """
        对单个子问题生成多个检索改写，分别检索并合并去重。

        返回值保留子问题、改写查询、每路检索结果和最终知识列表，供从智能体
        接收结构化任务包，而不是只接收丢失来源关系的知识字符串。
        """
        rewrites = rewrite_question_variants(
            sub_question,
            model,
            rewrite_count=rewrite_count,
        )
        retrieval_details = []
        knowledge = []
        seen_knowledge = set()

        for query in rewrites:
            try:
                result = self.retrieve_docs_for_question(query, self.sub_topk)
            except Exception as e:
                print(f"子问题检索失败（{query}）:", e)
                retrieval_details.append({
                    "query": query,
                    "knowledge": [],
                    "error": str(e),
                })
                continue
            docs = result.get("main_docs") or []
            query_knowledge = []

            for doc in docs:
                if not isinstance(doc, dict):
                    continue
                content = (doc.get("a") or "").strip()
                if not content:
                    continue
                query_knowledge.append(content)
                if content not in seen_knowledge:
                    seen_knowledge.add(content)
                    knowledge.append(content)

            retrieval_details.append({
                "query": query,
                "knowledge": query_knowledge,
            })

        return {
            "sub_question": sub_question,
            "rewrites": rewrites,
            "retrieval_details": retrieval_details,
            "knowledge": knowledge,
        }

    def retrieve_docs_multi_channel(self, question, model, is_polish=False,need_dim = None):
        main_results = self.retrieve_docs_for_question(question, self.main_topk) 
        print("question\n", question)
        print("main_results")
        # print(main_results)
        if need_dim == None:
            sub_questions = decompose_question(question, model)
        else:
            sub_questions = decompose_question_dim(question,model,need_dim)
        print("sub_questions")
        # print(sub_questions)
        all_docs = []
        if is_polish:
            sub_questions = [data['sub_question'] for data in sub_questions]
            sub_questions = judge_and_polish(sub_questions,model)
            print("polish_q")
            # print(sub_questions)
            for val in sub_questions:
                sub_results = self.retrieve_docs_for_question(val['refined_question'],self.sub_topk)     
                print("sub_question\n", val['refined_question'])
                print("sub_results")
                # print(sub_results)
                if sub_results["main_docs"] is not None:
                    all_docs.append({'sub_question':val['refined_question'],'retive_knowledge':[data['a'] for data in sub_results["main_docs"]]})
        else:
            for val in sub_questions:
                sub_results = self.retrieve_docs_for_question(val['sub_question'],self.sub_topk)   
                print("sub_question\n", val['sub_question']) 
                print("sub_results")
                # print(sub_results) 
                if sub_results["main_docs"] is not None:
                    all_docs.append({'sub_question':val['sub_question'],'retive_knowledge':[data['a'] for data in sub_results["main_docs"]]})
        # print("all_docs\n", all_docs)
        sub_results_all = []
        for doc in all_docs:
            question = doc['sub_question']
            for val in doc['retive_knowledge'][0:self.sub_topk]:
                if val not in sub_results_all:
                    save_docs = {
                        'q':question,
                        'a':val
                    }
                    sub_results_all.append(save_docs)
        # print("sub_results_all\n", sub_results_all)
        if len(sub_results_all) == 0:
            return {
                "main_docs": main_results["main_docs"],
                "sub_docs": None,
                "sub_questions": sub_questions
            }
        else:
            return {
                "main_docs": main_results["main_docs"],
                "sub_docs": sub_results_all,
                "sub_questions": sub_questions
            }
