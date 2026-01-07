import os
import time
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. 打印当前环境感知到的 HF_HOME
current_hf_home = os.environ.get("HF_HOME")
print(f"📋 当前环境变量 HF_HOME: {current_hf_home}")
if current_hf_home is None:
    print("   (使用默认路径: ~/.cache/huggingface/hub 或 C:\\Users\\用户名\\.cache\\huggingface\\hub)")
else:
    print(f"   (模型应该存储在: {os.path.join(current_hf_home, 'hub')})")

print("-" * 30)

# 2. 尝试加载模型
model_name = "BAAI/bge-reranker-base"
print(f"🚀 正在尝试加载模型: {model_name} ...")

try:
    start_time = time.time()
    # 这一步如果不报错，说明模型文件存在且完整
    model = HuggingFaceCrossEncoder(model_name=model_name)
    end_time = time.time()

    print(f"✅ 成功！模型加载耗时: {end_time - start_time:.2f} 秒")
    print("模型文件已正确下载且可被 Python 读取。")

except Exception as e:
    print(f"❌ 失败！模型加载出错: {e}")
    print("\n可能的解决方案：")
    print("1. 检查网络连接（是否需要代理）。")
    print("2. 检查 HF_HOME 路径下是否有写权限。")
    print("3. 手动删除缓存文件夹后重试。")