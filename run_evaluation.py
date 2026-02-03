"""
RAG 系统质量评估脚本
运行方式: python run_evaluation.py
"""

import json
import os
from src.rag_chain import get_rag_chain
from src.evaluation import RAGEvaluator, EvaluationSample, EvaluationResult

# ========== 配置 ==========
# 测试问题列表 (请根据你的知识库内容修改)
# ⚠️ 重要：问题必须与实际上传的文档内容相关！
# 通用问题如"总结文档"会导致检索失败

# 示例：针对 Terrapin Attack 论文的问题
TEST_QUESTIONS_TERRAPIN = [
    "什么是 Terrapin 攻击？它利用了什么漏洞？",
    "Terrapin 攻击如何影响 SSH 协议的安全性？",
    "如何防御 Terrapin 攻击？有哪些补丁措施？",
    "AsyncSSH 在 Terrapin 攻击中的漏洞是什么？",
]

# 示例：针对《数论基础》的问题
TEST_QUESTIONS_NUMBER_THEORY = [
    "数论基础这本书包含哪些章节内容？",
    "什么是除数函数 d(n)？它有什么性质？",
    "什么是原根和指标？",
]

# 当前使用的测试问题 (根据你上传的文档选择)
TEST_QUESTIONS = TEST_QUESTIONS_TERRAPIN + TEST_QUESTIONS_NUMBER_THEORY

# 如果有标准答案，可以添加 ground_truth
# TEST_DATA = [
#     {"question": "问题1", "ground_truth": "标准答案1"},
#     {"question": "问题2", "ground_truth": "标准答案2"},
# ]


def run_single_evaluation(chain, evaluator, question: str, ground_truth: str = None):
    """评估单个问题"""
    print(f"\n🔍 问题: {question}")
    
    # 调用 RAG
    result = chain.invoke({"input": question, "chat_history": []})
    answer = result["answer"]
    contexts = [doc.page_content for doc in result.get("source_documents", [])]
    
    print(f"💬 回答: {answer[:200]}..." if len(answer) > 200 else f"💬 回答: {answer}")
    
    # 评估
    sample = EvaluationSample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth
    )
    
    eval_result = evaluator.evaluate_sample(sample)
    
    print(f"📊 评分:")
    print(f"   Faithfulness: {eval_result.faithfulness:.2f}" if eval_result.faithfulness else "   Faithfulness: N/A")
    print(f"   Relevancy: {eval_result.answer_relevancy:.2f}" if eval_result.answer_relevancy else "   Relevancy: N/A")
    print(f"   Precision: {eval_result.context_precision:.2f}" if eval_result.context_precision else "   Precision: N/A")
    print(f"   综合: {eval_result.overall_score:.2f}" if eval_result.overall_score else "   综合: N/A")
    
    return eval_result


def run_batch_evaluation():
    """批量评估"""
    print("=" * 60)
    print("🚀 RAG 系统质量评估")
    print("=" * 60)
    
    # 初始化
    chain = get_rag_chain()
    if not chain:
        print("❌ RAG 链初始化失败，请先上传文档！")
        return
    
    evaluator = RAGEvaluator(llm=chain.llm)
    
    # 运行评估
    results = []
    for question in TEST_QUESTIONS:
        try:
            result = run_single_evaluation(chain, evaluator, question)
            results.append(result)
        except Exception as e:
            print(f"❌ 评估失败: {e}")
    
    # 汇总统计
    print("\n" + "=" * 60)
    print("📈 汇总统计")
    print("=" * 60)
    
    metrics = evaluator.get_aggregate_metrics()
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.2f}")
    
    # 保存详细报告
    report_path = "./logs/evaluation_report.json"
    os.makedirs("./logs", exist_ok=True)
    
    report = {
        "total_samples": len(results),
        "aggregate_metrics": metrics,
        "details": [r.to_dict() for r in results]
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 详细报告已保存到: {report_path}")


def interactive_evaluation():
    """交互式评估 - 逐个问题测试"""
    print("=" * 60)
    print("🔬 交互式 RAG 评估")
    print("输入问题进行评估，输入 'quit' 退出")
    print("=" * 60)
    
    chain = get_rag_chain()
    if not chain:
        print("❌ RAG 链初始化失败！")
        return
    
    evaluator = RAGEvaluator(llm=chain.llm)
    
    while True:
        question = input("\n📝 请输入问题: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        
        try:
            run_single_evaluation(chain, evaluator, question)
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print("\n👋 评估结束")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_evaluation()
    else:
        run_batch_evaluation()
