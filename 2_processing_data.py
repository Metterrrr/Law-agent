import re
import json
import os
import logging
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

#日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#仅提取法条，忽略其他层级
PATTERN_TIAO = re.compile(
    r"(?=第\s*[0-9零一二三四五六七八九十百千万亿]+\s*条(?:\s*之\s*[零一二三四五六七八九十百千\d]+)?\s*)"
)
# 用于提取纯净条号的正则
PATTERN_TIAO_TITLE = re.compile(
    r"第\s*[0-9零一二三四五六七八九十百千万亿]+\s*条(?:\s*之\s*[零一二三四五六七八九十百千\d]+)?"
)

def clean_text(text):
    """清洗文本：合并多空格、过滤特殊字符、保留有效内容"""
    if not text:
        return ""
    # 合并多余空格和换行
    text = re.sub(r"\s+", " ", text)
    # 过滤无效特殊字符（保留中文、英文、数字、常见标点）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。；：？！、（）【】《》\s]", "", text)
    return text.strip()

def skip_table_of_contents(content):
    """彻底跳过目录"""
    toc_patterns = [r"目\s*录", r"目\n录", r"目录", r"Contents"]
    for pattern in toc_patterns:
        toc_match = re.search(pattern, content, re.M | re.I)
        if toc_match:
            return content[toc_match.end():]
    return content

def split_and_extract_tiao(content):
    """只分割并提取条层级（修复版：无硬编码过滤、文本清洗）"""
    content = skip_table_of_contents(content)
    parts = PATTERN_TIAO.split(content)
    
    tiao_titles = []
    tiao_contents = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        title_match = PATTERN_TIAO_TITLE.match(part)
        if not title_match:
            continue
        
        # 提取纯净条号（去掉多余空格）
        tiao_title = re.sub(r"\s+", "", title_match.group())
        # 提取并清洗条内容
        tiao_content = clean_text(part[title_match.end():])
        
        if tiao_content:
            tiao_titles.append(tiao_title)
            tiao_contents.append(tiao_content)
    
    return tiao_titles, tiao_contents

def structure_law_document(law_doc):
    """只提取法律文档中的条"""
    content = law_doc.get("content", "")
    if not content:
        logger.warning(f"文档 {law_doc.get('title', '未知')} 无内容，跳过")
        return []
    
    tiao_titles, tiao_contents = split_and_extract_tiao(content)
    structured_chunks = []
    
    for title, content in zip(tiao_titles, tiao_contents):
        full_content = f"{title} {content}"
        structured_chunks.append({
            "doc_id": law_doc["doc_id"],
            "law_title": law_doc["title"],
            "bian": "",
            "zhang": "",
            "jie": "",
            "tiao": title,
            "content": full_content,
            "publish_date": law_doc.get("publish_date", ""),
            "effective_date": law_doc.get("effective_date", "")
        })
    
    return structured_chunks

def main():
    logger.info("="*60)
    logger.info("开始处理法律文档（仅提取条层级）")
    logger.info("="*60)
    
    #前置校验文件存在性
    input_file = os.path.join(RAW_DATA_DIR, "laws_raw.json")
    if not os.path.exists(input_file):
        logger.error(f"前置文件缺失：{input_file}，请先执行 1_lawsdata_reading.py")
        return
    
    with open(input_file, "r", encoding="utf-8") as f:
        laws = json.load(f)
    
    all_structured_chunks = []
    total_tiao = 0
    
    for i, law in enumerate(laws, 1):
        logger.info(f"\n[{i}/{len(laws)}] 正在处理: {law['title']}")
        try:
            chunks = structure_law_document(law)
            chunk_count = len(chunks)
            total_tiao += chunk_count
            all_structured_chunks.extend(chunks)
            logger.info(f"  ✅ 成功提取 {chunk_count} 个法条")
        except Exception as e:
            logger.error(f"  ❌ 处理失败 {law['title']}: {str(e)}", exc_info=True)
            continue
    
    # 保存结果
    output_file = os.path.join(PROCESSED_DATA_DIR, "laws_structured_only_tiao.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_structured_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info(f"✅ 处理完成！")
    logger.info(f"✅ 共处理 {len(laws)} 部法律")
    logger.info(f"✅ 总计提取 {total_tiao} 个法条")
    logger.info(f"✅ 数据已保存到: {output_file}")
    logger.info("="*60)
    logger.info("\n请继续执行: python 3_vector_database.py")

if __name__ == "__main__":
    main()