"""
RAGAS 评估框架模块
用于量化评估 RAG 系统的检索和生成质量

评估指标:
- Faithfulness: 答案是否基于上下文生成 (无幻觉)
- Answer Relevancy: 答案是否回答了用户问题
- Context Precision: 检索到的上下文有多少是真正有用的
- Context Recall: 是否检索到了所有必要的信息
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class EvaluationSample:
    """评估样本数据结构"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None  # 标准答案 (可选)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class EvaluationResult:
    """评估结果数据结构"""
    sample_id: str
    question: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    overall_score: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)


class RAGEvaluator:
    """
    RAG 评估器
    支持两种模式:
    1. 使用 RAGAS 库 (需要安装 ragas)
    2. 使用 LLM 自评估 (无额外依赖)
    """
    
    # LLM 自评估 Prompt 模板
    FAITHFULNESS_PROMPT = """你是一个评估专家。判断下面的【答案】是否完全基于【上下文】生成，没有添加上下文中不存在的信息。

【上下文】:
{context}

【答案】:
{answer}

评分标准:
- 1.0: 答案完全基于上下文，没有任何额外信息
- 0.7: 答案主要基于上下文，有少量合理推断
- 0.3: 答案部分基于上下文，但包含较多推测
- 0.0: 答案与上下文无关或存在明显错误

请只返回一个 0-1 之间的数字作为分数:"""

    RELEVANCY_PROMPT = """你是一个评估专家。判断下面的【答案】是否充分回答了【问题】。

【问题】:
{question}

【答案】:
{answer}

评分标准:
- 1.0: 答案完整、准确地回答了问题
- 0.7: 答案回答了问题的主要部分
- 0.3: 答案部分相关但不够完整
- 0.0: 答案与问题无关

请只返回一个 0-1 之间的数字作为分数:"""

    CONTEXT_PRECISION_PROMPT = """你是一个评估专家。判断下面的【检索上下文】中有多少内容对回答【问题】是真正有用的。

【问题】:
{question}

【检索上下文】:
{context}

评分标准:
- 1.0: 所有上下文都与问题高度相关
- 0.7: 大部分上下文相关，少量噪声
- 0.3: 只有部分上下文相关，较多噪声
- 0.0: 上下文与问题完全无关

请只返回一个 0-1 之间的数字作为分数:"""

    def __init__(self, llm=None, use_ragas: bool = False):
        """
        Args:
            llm: LangChain LLM 实例，用于 LLM 自评估模式
            use_ragas: 是否使用 RAGAS 库 (需要安装)
        """
        self.llm = llm
        self.use_ragas = use_ragas
        self.results_path = "./logs/evaluation_results.jsonl"
        
        if use_ragas:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness, 
                    answer_relevancy, 
                    context_precision,
                    context_recall
                )
                self.ragas_evaluate = evaluate
                self.ragas_metrics = [
                    faithfulness, 
                    answer_relevancy, 
                    context_precision
                ]
                print("✅ RAGAS 库已加载")
            except ImportError:
                print("⚠️ RAGAS 未安装，回退到 LLM 自评估模式")
                self.use_ragas = False
    
    def _extract_score(self, response_text: str) -> float:
        """从 LLM 响应中提取分数"""
        import re
        # 尝试匹配 0-1 之间的浮点数
        match = re.search(r'(0\.\d+|1\.0|1|0)', response_text.strip())
        if match:
            return float(match.group(1))
        return 0.5  # 默认中等分数
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """评估答案的忠实度 (是否基于上下文)"""
        if not self.llm:
            return None
        
        context_str = "\n\n".join(contexts)
        prompt = self.FAITHFULNESS_PROMPT.format(
            context=context_str[:3000],
            answer=answer
        )
        
        try:
            response = self.llm.invoke(prompt)
            score_text = response.content if hasattr(response, 'content') else str(response)
            return self._extract_score(score_text)
        except Exception as e:
            print(f"忠实度评估失败: {e}")
            return None
    
    def evaluate_relevancy(self, question: str, answer: str) -> float:
        """评估答案相关性 (是否回答了问题)"""
        if not self.llm:
            return None
        
        prompt = self.RELEVANCY_PROMPT.format(
            question=question,
            answer=answer
        )
        
        try:
            response = self.llm.invoke(prompt)
            score_text = response.content if hasattr(response, 'content') else str(response)
            return self._extract_score(score_text)
        except Exception as e:
            print(f"相关性评估失败: {e}")
            return None
    
    def evaluate_context_precision(self, question: str, contexts: List[str]) -> float:
        """评估上下文精度 (检索的信噪比)"""
        if not self.llm:
            return None
        
        context_str = "\n\n".join(contexts)
        prompt = self.CONTEXT_PRECISION_PROMPT.format(
            question=question,
            context=context_str[:3000]
        )
        
        try:
            response = self.llm.invoke(prompt)
            score_text = response.content if hasattr(response, 'content') else str(response)
            return self._extract_score(score_text)
        except Exception as e:
            print(f"上下文精度评估失败: {e}")
            return None
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """评估单个样本"""
        import uuid
        
        sample_id = str(uuid.uuid4())[:8]
        
        if self.use_ragas:
            # 使用 RAGAS 评估
            try:
                from datasets import Dataset
                
                dataset = Dataset.from_dict({
                    "question": [sample.question],
                    "answer": [sample.answer],
                    "contexts": [sample.contexts],
                    "ground_truth": [sample.ground_truth] if sample.ground_truth else [sample.answer]
                })
                
                results = self.ragas_evaluate(dataset, metrics=self.ragas_metrics)
                
                return EvaluationResult(
                    sample_id=sample_id,
                    question=sample.question,
                    faithfulness=results.get("faithfulness"),
                    answer_relevancy=results.get("answer_relevancy"),
                    context_precision=results.get("context_precision"),
                    overall_score=sum(filter(None, [
                        results.get("faithfulness"),
                        results.get("answer_relevancy"),
                        results.get("context_precision")
                    ])) / 3
                )
            except Exception as e:
                print(f"RAGAS 评估失败: {e}，回退到 LLM 自评估")
        
        # LLM 自评估模式
        faithfulness = self.evaluate_faithfulness(sample.answer, sample.contexts)
        relevancy = self.evaluate_relevancy(sample.question, sample.answer)
        precision = self.evaluate_context_precision(sample.question, sample.contexts)
        
        scores = [s for s in [faithfulness, relevancy, precision] if s is not None]
        overall = sum(scores) / len(scores) if scores else None
        
        result = EvaluationResult(
            sample_id=sample_id,
            question=sample.question,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            overall_score=overall
        )
        
        # 保存结果
        self._save_result(result)
        
        return result
    
    def evaluate_batch(self, samples: List[EvaluationSample]) -> List[EvaluationResult]:
        """批量评估"""
        results = []
        for i, sample in enumerate(samples):
            print(f"评估样本 {i+1}/{len(samples)}...")
            result = self.evaluate_sample(sample)
            results.append(result)
        return results
    
    def _save_result(self, result: EvaluationResult):
        """保存评估结果"""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """获取聚合指标"""
        if not os.path.exists(self.results_path):
            return {}
        
        metrics = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "overall_score": []
        }
        
        with open(self.results_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    for key in metrics:
                        if result.get(key) is not None:
                            metrics[key].append(result[key])
                except:
                    continue
        
        return {
            key: sum(values) / len(values) if values else None
            for key, values in metrics.items()
        }


def create_evaluation_dataset_from_logs(log_path: str = "./logs/rag_activity.jsonl") -> List[EvaluationSample]:
    """
    从 RAG 活动日志创建评估数据集
    """
    samples = []
    
    if not os.path.exists(log_path):
        print(f"日志文件不存在: {log_path}")
        return samples
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                log = json.loads(line)
                # 需要从日志中提取上下文信息
                # 这里简化处理，实际使用时需要根据日志格式调整
                sample = EvaluationSample(
                    question=log.get("question", ""),
                    answer=log.get("answer", ""),
                    contexts=[],  # 需要从其他来源获取
                    ground_truth=None
                )
                if sample.question and sample.answer:
                    samples.append(sample)
            except:
                continue
    
    return samples


# 便捷函数
def quick_evaluate(question: str, answer: str, contexts: List[str], llm=None) -> EvaluationResult:
    """
    快速评估单个 RAG 响应
    
    Usage:
        from src.evaluation import quick_evaluate
        result = quick_evaluate(
            question="什么是机器学习？",
            answer="机器学习是...",
            contexts=["文档片段1", "文档片段2"],
            llm=your_llm
        )
        print(f"整体分数: {result.overall_score}")
    """
    evaluator = RAGEvaluator(llm=llm)
    sample = EvaluationSample(
        question=question,
        answer=answer,
        contexts=contexts
    )
    return evaluator.evaluate_sample(sample)
