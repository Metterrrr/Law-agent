import json
import os
import hashlib 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import (
    PROCESSED_DATA_DIR,
    VECTOR_DB_DIR,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LAW_MATCH_TOP_N,
    RAW_DATA_DIR,
)

def get_embedding():
    return OllamaEmbeddings(
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL
    )

def get_collection_name(doc_id: str) -> str:
    """生成符合ChromaDB要求的集合名称：使用MD5哈希"""
    hash_obj = hashlib.md5(doc_id.encode('utf-8'))
    hash_str = hash_obj.hexdigest()
    return f"law_{hash_str}"

# 在build_law_catalog函数中，优化法律目录的内容
def build_law_catalog(laws):
    print("\n正在构建法律目录向量库...")
    catalog_docs = []
    
    # 法律类型映射
    law_type_map = {
            # 宪法及宪法相关法
        "宪法": "宪法及宪法相关法",
        "立法法": "宪法及宪法相关法",
        "全国人民代表大会和地方各级人民代表大会选举法": "宪法及宪法相关法",
        "民族区域自治法": "宪法及宪法相关法",
        "香港特别行政区基本法": "宪法及宪法相关法",
        "澳门特别行政区基本法": "宪法及宪法相关法",
        "国旗法": "宪法及宪法相关法",
        "国徽法": "宪法及宪法相关法",
        "国籍法": "宪法及宪法相关法",
    
            # 刑事法律
        "刑法": "刑事法律",
        "刑事诉讼法": "刑事法律",
        "监狱法": "刑事法律",
    
            # 民事法律
        "民法典": "民事法律",
        "民事诉讼法": "民事法律",
        "著作权法": "民事法律",
        "专利法": "民事法律",
        "商标法": "民事法律",
    
            # 行政法律
        "行政法": "行政法律",
        "行政诉讼法": "行政法律",
        "行政处罚法": "行政法律",
        "行政复议法": "行政法律",
        "行政强制法": "行政法律",
        "国家赔偿法": "行政法律",
        "公务员法": "行政法律",
    
            # 劳动法律
        "劳动法": "劳动法律",
        "劳动合同法": "劳动法律",
        "劳动争议调解仲裁法": "劳动法律",
    
            # 交通法律
        "道路交通安全法": "交通法律",
        "铁路法": "交通法律",
        "民用航空法": "交通法律",
        "海上交通安全法": "交通法律",
    
        # 经济法律
        "反垄断法": "经济法律",
        "反不正当竞争法": "经济法律",
        "消费者权益保护法": "经济法律",
        "产品质量法": "经济法律",
        "企业所得税法": "经济法律",
        "个人所得税法": "经济法律",
        "税收征收管理法": "经济法律",
        "商业银行法": "经济法律",
        "银行业监督管理法": "经济法律",
    
            # 商事法律
        "公司法": "商事法律",
        "合伙企业法": "商事法律",
        "个人独资企业法": "商事法律",
        "证券法": "商事法律",
        "保险法": "商事法律",
        "票据法": "商事法律",
        "企业破产法": "商事法律",
        "海商法": "商事法律",
    
            # 社会法律
        "未成年人保护法": "社会法律",
        "老年人权益保障法": "社会法律",
        "妇女权益保障法": "社会法律",
        "残疾人保障法": "社会法律",
        "社会保险法": "社会法律",
        "社会救助法": "社会法律",
        "慈善法": "社会法律",
    
            # 环境资源法律
        "环境保护法": "环境资源法律",
        "水污染防治法": "环境资源法律",
        "大气污染防治法": "环境资源法律",
        "土壤污染防治法": "环境资源法律",
        "森林法": "环境资源法律",
        "草原法": "环境资源法律",
        "矿产资源法": "环境资源法律",
        "水法": "环境资源法律",
        "野生动物保护法": "环境资源法律",
    
            # 教育文化法律
        "教育法": "教育文化法律",
        "高等教育法": "教育文化法律",
        "义务教育法": "教育文化法律",
        "职业教育法": "教育文化法律",
        "文物保护法": "教育文化法律",
        "非物质文化遗产法": "教育文化法律",
    
            # 卫生健康法律
        "食品安全法": "卫生健康法律",
        "药品管理法": "卫生健康法律",
        "传染病防治法": "卫生健康法律",
        "基本医疗卫生与健康促进法": "卫生健康法律"
    }
    
    for law in laws:
        law_title = law["title"]
        # 自动识别法律类型
        law_type = "其他法律"
        for key, value in law_type_map.items():
            if key in law_title:
                law_type = value
                break
        
        # 优化目录内容：类型+标题+前200字
        catalog_content = f"【{law_type}】{law_title}。核心内容：{law['content'][:300]}"
        
        catalog_doc = Document(
            page_content=catalog_content,
            metadata={
                "doc_id": law["doc_id"],
                "law_title": law["title"],
                "law_type": law_type,  
                "publish_date": law.get("publish_date", ""),
                "effective_date": law.get("effective_date", "")
            }
        )
        catalog_docs.append(catalog_doc)
    
    embeddings = get_embedding()
    catalog_db = Chroma.from_documents(
        documents=catalog_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name="law_catalog"
    )
    print(f"✅ 法律目录库构建完成，共收录 {len(catalog_docs)} 部法律")
    return catalog_db

def build_individual_law_dbs(laws):
    embeddings = get_embedding()
    total_laws = len(laws)

    # 【修复】只加载1次，不在循环里读
    input_file = os.path.join(PROCESSED_DATA_DIR, "laws_structured_only_tiao.json")
    with open(input_file, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    for i, law in enumerate(laws, 1):
        doc_id = law["doc_id"]
        law_title = law["title"]
        collection_name = get_collection_name(doc_id)

        print(f"\n[{i}/{total_laws}] 正在构建: {law_title}")
        print(f"  集合名称: {collection_name}")

        # 筛选当前法律
        law_chunks = [c for c in all_chunks if c["doc_id"] == doc_id]

        if not law_chunks:
            print(f"  ⚠️ 未找到法条，跳过")
            continue

        documents = []
        for chunk in law_chunks:
            doc = Document(
                page_content=chunk.get("content", ""),
                metadata={
                    "doc_id": chunk.get("doc_id", ""),
                    "law_title": chunk.get("law_title", ""),
                    "bian": chunk.get("bian", ""),
                    "zhang": chunk.get("zhang", ""),
                    "jie": chunk.get("jie", ""),
                    "tiao": chunk.get("tiao", ""),
                    "effective_date": chunk.get("effective_date", "")
                }
            )
            documents.append(doc)

        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR,
            collection_name=collection_name
        )
        print(f"  ✅ 成功构建，包含 {len(documents)} 个法条")

def main():
    print("="*60)
    print("开始构建分库式法律向量数据库")
    print("="*60)
    
    # 加载原始法律数据
    input_file = os.path.join(RAW_DATA_DIR, "laws_raw.json")
    with open(input_file, "r", encoding="utf-8") as f:
        laws = json.load(f)
    
    print(f"加载了 {len(laws)} 部法律")
    
    # 第一步：构建法律目录库
    build_law_catalog(laws)
    
    # 第二步：为每部法律构建独立库
    build_individual_law_dbs(laws)
    
    print("\n" + "="*60)
    print("✅ 所有向量数据库构建完成！")
    print(f"✅ 数据库保存位置: {VECTOR_DB_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()