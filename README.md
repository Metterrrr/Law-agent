法律助手 RAG 系统
基于**分库式向量检索**的本地中国法律问答助手，采用LangChain+Chroma+Ollama+DeepSeek技术栈，实现法条匹配与专业结构化法律解答。

 ✨ 核心功能
- 📄 **批量文档处理**：自动读取本地docx格式法律文件，支持大文件解析与异常处理
- 🧩 **智能法条分割**：基于正则的精准条级拆分，自动跳过目录与无效内容，文本清洗去噪
- 🗄️ **分库向量架构**：法律目录库+单法律独立库，大幅提升检索效率与准确率
- 🏷️ **智能法律分类**：自动识别10大类法律（宪法、刑事、民事、行政、劳动等）
- 🔍 **增强检索引擎**：混合语义+关键词重排 + 自动查询重写（Step-back/HyDE/复合策略）
- 📊 **检索质量评估**：AI自动评估检索结果相关性，自动触发查询扩展
- 💬 **流式对话输出**：支持SSE流式响应，实时展示回答过程
- 📚 **完整会话管理**：多会话历史保存、查询与删除
- 🎯 **问题分类处理**：自动区分法律/非法律问题，适配不同回答模式
- 🔗 **法条溯源**：回答自动标注引用的具体法律条文

## 🛠️ 技术栈
| 模块 | 技术选型 |
|------|----------|
| 语言 | Python 3.12 |
| 向量数据库 | ChromaDB 0.5+ |
| 嵌入模型 | Ollama (nomic-embed-text) |
| 大语言模型 | DeepSeek API (deepseek-chat) |
| 框架 | LangChain 0.2+, FastAPI 0.110+ |
| 前端 | 原生HTML/JavaScript |
| 文档解析 | python-docx 1.1+ |
| 中文分词 | jieba 0.42.1 |

## 🚀 快速开始
### 1. 环境准备
- 安装Git：https://git-scm.com/
- 安装Ollama：https://ollama.com/
- 拉取嵌入模型：
  ```bash
  ollama pull nomic-embed-text
  ```
- 注册DeepSeek账号并获取API密钥：https://platform.deepseek.com/

### 2. 克隆项目并安装依赖
```bash
git clone https://github.com/你的用户名/Law-agent.git
cd Law-agent
pip install -r requirements.txt
```

### 3. 配置环境变量
复制项目根目录的`.env.example`为`.env`，填写你的DeepSeek API密钥：
```env
# DeepSeek API配置
DEEPSEEK_API_KEY=你的DeepSeek_API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Ollama嵌入配置
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
#请激活虚拟环境后进行接下来的操作
```


### 4. 准备法律文档
先运行config.py,生成相应文件夹后，
将所有docx格式的法律文件放入项目根目录的`law_data/`文件夹中。

### 5. 构建向量数据库
**严格按顺序执行以下脚本**：
```bash
# 1. 读取本地法律文档并生成原始JSON数据
python 1_lawsdata_reading.py

# 2. 结构化处理，提取并清洗法条
python 2_processing_data.py

# 3. 构建分库式向量数据库（目录库+单法律库）
python 3_vector_database.py
```

### 6. 启动Web服务
```bash
python web_server.py
```
服务启动后会自动打开浏览器访问 `http://127.0.0.1:8000`，即可开始使用。

## 📁 项目结构
```
Law-agent/bot
├── 1_lawsdata_reading.py    # 本地docx文档读取模块
├── 2_processing_data.py     # 法条分割、清洗与结构化处理
├── 3_vector_database.py     # 分库式向量数据库构建
├── config.py                # 全局配置文件（已支持.env环境变量）
├── search.py                # RAG检索引擎核心（重排、查询重写、相关性评估）
├── law_agent.py             # 法律对话Agent与会话管理
├── web_server.py            # FastAPI后端服务
├── web_frontend.html        # 前端交互页面
├── requirements.txt         # Python依赖列表
└── README.md                # 项目说明文档
```

## 📖 使用说明
1. **法律问答**：在输入框中输入法律问题，系统会自动：
   - 判断是否为法律问题
   - 检索最相关的法律
   - 提取相关法条并进行重排
   - 按"核心分析→检索过程→推理过程→最终结论"四步输出结构化回答

2. **回答格式**：所有法律问题均采用统一的四步式输出，确保逻辑清晰、专业严谨
3. **会话管理**：左侧面板可查看历史会话、切换会话或删除会话
4. **检索溯源**：回答末尾会显示引用的具体法律条文，支持点击查看完整内容

## ❗ 常见问题
1. **Ollama连接失败**：请确保Ollama服务已启动，且`EMBEDDING_BASE_URL`配置正确
2. **未检索到相关法条**：检查`law_data/`目录是否有docx文件，或重新运行向量库构建脚本
3. **API调用失败**：确认DeepSeek API密钥正确且账户有足够余额
4. **向量库构建缓慢**：这是正常现象，取决于法律文件数量和电脑性能，请耐心等待
5. **法条分割不准确**：可以修改`2_processing_data.py`中的正则表达式以适配特定文档格式

## ⚠️ 免责声明
本项目仅供学习和参考使用，**不构成任何法律意见**。所有法律问题请咨询专业律师，本项目作者不承担因使用本系统产生的任何法律责任。



## 🤝 贡献
欢迎提交Issue和Pull Request来改进本项目！

---

**如果本项目对你有帮助，欢迎Star ⭐**
