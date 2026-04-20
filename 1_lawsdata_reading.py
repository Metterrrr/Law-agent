import os
import json
import logging
from datetime import datetime
from docx import Document
from config import RAW_DATA_DIR, LOCAL_LAW_DIR

#日志系统 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_local_docx(file_path):
    """读取本地docx文件内容"""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        #增强异常日志
        logger.error(f"读取文件失败 {file_path}: {str(e)}", exc_info=True)
        return None

def load_local_laws():
    """批量读取本地docx法律文件"""
    all_laws = []
    
    os.makedirs(LOCAL_LAW_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 遍历所有docx文件
    for filename in os.listdir(LOCAL_LAW_DIR):
        if not filename.endswith(".docx"):
            continue
        law_name = os.path.splitext(filename)[0]
        file_path = os.path.join(LOCAL_LAW_DIR, filename)
        
        logger.info(f"正在读取: {law_name}")
        content = read_local_docx(file_path)
        
        if not content:
            logger.warning(f"文件 {law_name} 内容为空，跳过")
            continue
        
        law_data = {
            "doc_id": f"local_{law_name}",
            "title": law_name,
            "publish_date": "待定（需从文档提取）",
            "effective_date": "待定（需从文档提取）",
            "content": content,
            "source": "本地文件",
            "crawl_time": datetime.now().isoformat()
        }
        all_laws.append(law_data)
    return all_laws

def main():
    logger.info("="*60)
    logger.info("开始读取本地法律文档")
    logger.info("="*60)
    
    all_laws = load_local_laws()
    
    if not all_laws:
        logger.error("未读取到任何有效法律文档，请检查 LOCAL_LAW_DIR 路径")
        return
    
    output_file = os.path.join(RAW_DATA_DIR, "laws_raw.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=2)
    
    logger.info("="*60)
    logger.info(f"✅ 读取完成！共加载 {len(all_laws)} 部法律")
    logger.info(f"✅ 数据已保存到: {output_file}")
    logger.info("="*60)

if __name__ == "__main__":
    main()