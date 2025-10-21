## `backend/pipeline.py`（基于之前提供的完整脚本，做了少量适配，保留 deepseek 占位）
# 结构化文本提取管道 - 用于从医疗文本中提取结构化信息

import re
import json
import os
import csv
import logging
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 基础路径配置
BASE_DIR = os.path.dirname(__file__)  # 当前文件所在目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')  # 输出文件目录
RULES_PATH = os.path.join(BASE_DIR, 'rules.json')  # 规则配置文件路径
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

# 配置日志系统
# 配置日志输出路径、级别为INFO，定义日志格式：时间戳 + 日志级别 + 消息内容
# 使用更可靠的日志配置方法避免乱码
log_file = os.path.join(OUTPUT_DIR, 'pipeline.log')

# 创建文件处理器并设置UTF-8编码
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# DeepSeek API 配置
DEEPSEEK_ENABLE = True  # DeepSeek API 开关

# 从环境变量获取API密钥，如果没有设置则使用None
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
# 检查API密钥是否有效（不以'sk-'开头或长度不足的密钥视为无效）
if DEEPSEEK_API_KEY and (not DEEPSEEK_API_KEY.startswith('sk-') or len(DEEPSEEK_API_KEY) < 20):
    logger.warning(f"检测到无效的DeepSeek API密钥格式，将禁用DeepSeek功能")
    DEEPSEEK_API_KEY = None

DEEPSEEK_ENDPOINT = os.environ.get('DEEPSEEK_ENDPOINT', 'https://api.deepseek.com')  # DeepSeek API 端点

# 默认解析规则 - 用于医疗文本中的关键参数提取
DEFAULT_RULES = [
    {
        "name": "LVEF",  # 参数名称：左室射血分数
        "keywords": ["LVEF", "射血分数", "左室射血分数"],  # 关键词列表
        "regex": r"(LVEF[:=]?\s*([0-9]{1,3}\s*%?))|(射血分数[:：]?\s*([0-9]{1,3}\s*%?))"  # 正则表达式模式
    },
    {
        "name": "左室收缩功能",  # 参数名称：左室收缩功能
        "keywords": ["左室收缩功能", "收缩功能", "左室收缩力", "心室肌收缩力"],  # 关键词列表
        "regex": r"(左室收缩功能(?:\s*[:：]?\s*)(降低|减弱|正常|减低|减低了|差|下降))|(收缩力(?:\s*[:：]?\s*)(降低|减弱|正常|下降|差))"  # 正则表达式模式
    }
]

# 确保规则配置文件存在，如果不存在则创建默认规则
if not os.path.exists(RULES_PATH):
    with open(RULES_PATH, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_RULES, f, ensure_ascii=False, indent=2)


def load_rules(path: str = RULES_PATH) -> List[Dict[str, Any]]:
    """
    从指定路径加载解析规则
    
    Args:
        path: 规则文件路径，默认为 RULES_PATH
        
    Returns:
        List[Dict[str, Any]]: 解析规则列表
    """
    # 确保规则文件存在，如果不存在则创建默认规则
    if not os.path.exists(path):
        logger.info(f"规则文件 {path} 不存在，创建默认规则")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_RULES, f, ensure_ascii=False, indent=2)
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_structured_to_json(data: List[Dict[str, Any]], path: str = os.path.join(OUTPUT_DIR, 'structured.json')):
    """
    将结构化数据保存为 JSON 文件
    
    Args:
        data: 要保存的结构化数据列表
        path: 输出文件路径，默认为 OUTPUT_DIR/structured.json
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_structured_to_csv(data: List[Dict[str, Any]], path: str = os.path.join(OUTPUT_DIR, 'structured.csv')):
    """
    将结构化数据保存为 CSV 文件
    
    Args:
        data: 要保存的结构化数据列表
        path: 输出文件路径，默认为 OUTPUT_DIR/structured.csv
    """
    fieldnames = ['doc_id', 'raw_text', 'extracted_json', 'status', 'line_results_json']  # CSV 文件列名
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头
        for d in data:
            writer.writerow({
                'doc_id': d.get('doc_id'),  # 文档ID
                'raw_text': d.get('raw_text'),  # 原始文本
                'extracted_json': json.dumps(d.get('extracted', []), ensure_ascii=False),  # 提取结果（JSON格式）
                'status': d.get('status'),  # 处理状态
                'line_results_json': json.dumps(d.get('line_results', []), ensure_ascii=False)  # 逐行处理结果（JSON格式）
            })


def log_failure(doc_id: str, raw_text: str, reason: str, path: str = os.path.join(OUTPUT_DIR, 'failures.jsonl')):
    """
    记录处理失败的文档信息
    
    Args:
        doc_id: 文档ID
        raw_text: 原始文本内容
        reason: 失败原因
        path: 失败记录文件路径，默认为 OUTPUT_DIR/failures.jsonl
    """
    entry = {'doc_id': doc_id, 'raw_text': raw_text, 'reason': reason}  # 失败记录条目
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')  # 以 JSONL 格式追加写入
    logger.error(f"Failure: {doc_id} reason={reason}")  # 记录错误日志


def parse_with_rules(text: str, rules: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    使用预定义规则从文本中提取结构化信息
    
    Args:
        text: 要解析的文本内容
        rules: 解析规则列表
        
    Returns:
        List[Dict[str, str]]: 提取的参数列表，每个参数包含名称和值
    """
    res = []  # 存储提取结果的列表
    
    # 遍历所有规则
    for r in rules:
        found = False  # 标记是否在当前规则中找到匹配
        regex = r.get('regex')  # 获取正则表达式规则
        
        # 首先尝试使用正则表达式匹配
        if regex:
            m = re.search(regex, text, flags=re.I)  # 忽略大小写进行匹配
            if m:
                groups = [g for g in m.groups() if g is not None]  # 获取非空匹配组
                if groups:
                    value = None
                    # 从后向前查找包含数字或百分比的组
                    for g in reversed(groups):
                        if isinstance(g, str) and re.search(r"[0-9％%]", g):
                            value = g.strip()
                            break
                    if not value:
                        value = groups[-1].strip()  # 使用最后一个非空组
                    res.append({'param_name': r['name'], 'param_value': value})
                else:
                    res.append({'param_name': r['name'], 'param_value': m.group(0).strip()})
                found = True
        
        # 如果正则表达式未匹配，尝试关键词匹配
        if not found:
            for kw in r.get('keywords', []):
                idx = text.lower().find(kw.lower())  # 不区分大小写查找关键词
                if idx != -1:
                    # 获取关键词后40个字符的内容进行进一步匹配
                    tail = text[idx: idx + len(kw) + 40]
                    m2 = re.search(r"(降低|减弱|正常|减低|下降|[0-9]{1,3}\s*%|[0-9]{1,3}％|四成|40%|38%)", tail, flags=re.I)
                    if m2:
                        res.append({'param_name': r['name'], 'param_value': m2.group(0).strip()})
                        found = True
                        break
    return res


def call_deepseek_extract(text: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """
    调用 DeepSeek API 进行文本提取
    
    Args:
        text: 要提取的文本内容
        system_prompt: 系统提示词（可选）
        
    Returns:
        List[Dict[str, str]]: 提取的参数列表
    """
    # 检查DeepSeek功能是否可用
    if not DEEPSEEK_ENABLE:
        logger.info("DeepSeek API 未启用")
        return []
    
    if not DEEPSEEK_API_KEY:
        logger.warning("DeepSeek API 密钥未配置，请设置 DEEPSEEK_API_KEY 环境变量")
        return []
    
    # 构建系统提示词
    if system_prompt is None:
        system_prompt = """你是一个专业的医疗文本分析助手。请从医疗文本中提取关键参数信息。
        请识别并提取以下类型的参数：
        - LVEF（左室射血分数）：数值百分比，如 45%、60% 等
        - 左室收缩功能：描述性状态，如 降低、减弱、正常、下降等
        - PASP（肺动脉收缩压）：数值，如 48mmHg、60mmHg 等
        - 其他医疗参数
        
        请以 JSON 格式返回结果，格式为：[{"param_name": "参数名", "param_value": "参数值"}, ...]"""
    
    # 构建用户消息
    user_message = f"请从以下医疗文本中提取关键参数信息：\n\n{text}"
    
    try:
        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_ENDPOINT
        )
        
        # 发送请求
        logger.info(f"调用 DeepSeek API 进行文本提取，文本长度：{len(text)}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        
        # 尝试从响应中提取 JSON 数据
        try:
            # 查找 JSON 格式的内容
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
                logger.info(f"DeepSeek API 提取成功，提取到 {len(extracted_data)} 个参数")
                return extracted_data
            else:
                # 如果无法解析为 JSON，尝试手动解析
                logger.warning(f"DeepSeek API 响应无法解析为 JSON: {content}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"DeepSeek API 响应 JSON 解析错误: {e}")
            return []
            
    except Exception as e:
        # 特别处理常见API错误
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            logger.error(f"DeepSeek API 认证失败：请检查 API 密钥是否正确配置")
            logger.error(f"错误详情：{error_msg}")
        elif "402" in error_msg or "insufficient balance" in error_msg.lower() or "余额不足" in error_msg:
            logger.error(f"DeepSeek API 余额不足：请为您的账户充值")
            logger.error(f"错误详情：{error_msg}")
        else:
            logger.error(f"DeepSeek API 调用异常: {error_msg}")
        return []


def process_documents(documents: List[Dict[str, str]], rules: List[Dict[str, Any]]):
    """
    批量处理文档，逐行提取结构化信息
    
    Args:
        documents: 文档列表，每个文档包含 doc_id 和 raw_text
        rules: 解析规则列表
        
    Returns:
        List[Dict[str, Any]]: 处理结果列表，包含逐行提取信息和状态
    """
    results = []  # 存储处理结果的列表
    
    # 遍历所有文档
    for doc in documents:
        doc_id = doc.get('doc_id')  # 获取文档ID
        text = doc.get('raw_text', '')  # 获取原始文本内容
        
        try:
            # 将文本按行分割
            lines = text.split('\n')
            line_results = []  # 存储每行的提取结果
            
            # 逐行处理文本
            for line_num, line in enumerate(lines, 1):
                line = line.strip()  # 去除首尾空白字符
                if not line:  # 跳过空行
                    continue
                    
                # 对每行文本进行结构化提取
                extracted = parse_with_rules(line, rules)
                
                # 如果规则解析失败，尝试使用 DeepSeek API
                if not extracted:
                    extracted = call_deepseek_extract(line)
                
                # 记录每行的提取结果
                line_result = {
                    'line_number': line_num,
                    'line_text': line,
                    'extracted': extracted,
                    'status': 'ok' if extracted else 'no_match'
                }
                line_results.append(line_result)
            
            # 汇总文档的提取结果
            all_extracted = []
            for line_result in line_results:
                all_extracted.extend(line_result['extracted'])
            
            # 如果整个文档都没有提取到任何信息，记录失败
            if not all_extracted:
                reason = 'no_extraction'  # 提取失败原因
                log_failure(doc_id, text, reason)  # 记录失败
                results.append({
                    'doc_id': doc_id, 
                    'raw_text': text, 
                    'extracted': [], 
                    'status': 'failed',
                    'line_results': line_results  # 包含逐行处理结果
                })
            else:
                # 提取成功，记录成功结果
                results.append({
                    'doc_id': doc_id, 
                    'raw_text': text, 
                    'extracted': all_extracted, 
                    'status': 'ok',
                    'line_results': line_results  # 包含逐行处理结果
                })
                
        except Exception as e:
            # 处理过程中发生异常，记录异常信息
            logger.exception(f"Exception processing doc {doc_id}")
            log_failure(doc_id, text, f"exception:{str(e)}")
            results.append({'doc_id': doc_id, 'raw_text': text, 'extracted': [], 'status': 'failed'})
    
    # 保存处理结果到文件
    save_structured_to_json(results)  # 保存为 JSON 格式
    save_structured_to_csv(results)   # 保存为 CSV 格式
    
    return results


def read_annotations(path: str = os.path.join(OUTPUT_DIR, 'annotations.csv')) -> List[Dict[str, str]]:
    """
    读取人工标注数据
    
    Args:
        path: 标注文件路径，默认为 OUTPUT_DIR/annotations.csv
        
    Returns:
        List[Dict[str, str]]: 标注数据列表
    """
    if not os.path.exists(path):
        return []  # 文件不存在时返回空列表
    
    anns = []  # 存储标注数据的列表
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # 创建 CSV 字典读取器
        for row in reader:
            anns.append(row)  # 逐行读取标注数据
    return anns


def assemble_deepseek_payload_for_rules(annotations: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    为 DeepSeek API 组装生成规则的请求载荷
    
    Args:
        annotations: 人工标注数据列表
        
    Returns:
        Dict[str, Any]: DeepSeek API 请求载荷
    """
    examples = []  # 存储示例数据
    
    # 使用前20个标注作为示例（避免请求过大）
    for a in annotations[:20]:
        examples.append({
            'text': a['raw_text'],  # 原始文本
            'label': {a['param_name']: a['param_value']}  # 标注标签
        })
    
    # 组装请求载荷
    payload = {
        'task': 'generate_extraction_rules',  # 任务类型
        'examples': examples,  # 示例数据
        'output_format': 'json_rules',  # 输出格式
        'instructions': '基于以上人工标注，生成一组 JSON 解析规则（name/keywords/regex）来覆盖这类情况。输出必须为 JSON 数组。'
    }
    return payload


def call_deepseek_generate_rules(annotations: List[Dict[str, str]]) -> Optional[List[Dict[str, Any]]]:
    """
    调用 DeepSeek API 生成新的解析规则
    
    Args:
        annotations: 人工标注数据列表
        
    Returns:
        Optional[List[Dict[str, Any]]]: 生成的规则列表，如果 DeepSeek 未启用则返回 None
    """
    if not DEEPSEEK_ENABLE:
        logger.info("DeepSeek API未启用，返回空规则列表")
        return None
    
    if not DEEPSEEK_API_KEY:
        logger.warning("DeepSeek API密钥未配置，请设置 DEEPSEEK_API_KEY 环境变量")
        return None
    
    if not annotations:
        logger.warning('没有标注数据可用于生成规则')
        return None
    
    try:
        # 初始化OpenAI客户端
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_ENDPOINT
        )
        
        # 构建系统提示词
        system_prompt = """你是一个专业的规则生成助手。请基于提供的标注数据生成 JSON 格式的解析规则。
        
        规则格式要求：
        {
            "name": "参数名称",
            "keywords": ["关键词1", "关键词2", ...],
            "regex": "正则表达式模式"
        }
        
        规则生成原则：
        1. 基于标注数据中的参数名称和值模式生成
        2. 关键词应覆盖参数的各种表达方式
        3. 正则表达式应能准确匹配参数值
        4. 规则应具有通用性，能处理类似情况
        
        请返回 JSON 数组格式的规则列表。"""
        
        # 构建用户消息
        examples_text = "\n".join([
            f"文本: {a['raw_text']} -> 参数: {a['param_name']} = {a['param_value']}"
            for a in annotations[:10]  # 使用前10个示例避免过长
        ])
        
        user_message = f"""请基于以下标注数据生成解析规则：
        
        {examples_text}
        
        请生成适用于这些标注数据的 JSON 规则数组。"""
        
        # 发送请求
        logger.info(f"调用 DeepSeek API 生成规则，标注数据数量：{len(annotations)}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        # 解析响应
        content = response.choices[0].message.content
        
        # 尝试从响应中提取 JSON 数据
        try:
            # 查找 JSON 格式的内容
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                generated_rules = json.loads(json_match.group())
                logger.info(f"DeepSeek API 规则生成成功，生成 {len(generated_rules)} 条规则")
                return generated_rules
            else:
                logger.warning(f"DeepSeek API 规则生成响应无法解析为 JSON: {content}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"DeepSeek API 规则生成响应 JSON 解析错误: {e}")
            return None
            
    except Exception as e:
        # 特别处理常见API错误
        error_msg = str(e)
        if "401" in error_msg or "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            logger.error(f"DeepSeek API 认证失败：请检查 API 密钥是否正确配置")
            logger.error(f"错误详情：{error_msg}")
        elif "402" in error_msg or "insufficient balance" in error_msg.lower() or "余额不足" in error_msg:
            logger.error(f"DeepSeek API 余额不足：请为您的账户充值")
            logger.error(f"错误详情：{error_msg}")
        else:
            logger.error(f"DeepSeek API 规则生成调用异常: {error_msg}")
        return None


def apply_new_rules(new_rules: List[Dict[str, Any]]):
    """
    应用新的解析规则并保存到配置文件
    
    Args:
        new_rules: 新的解析规则列表
    """
    if not new_rules:
        return  # 如果新规则为空，直接返回
    
    # 将新规则写入配置文件
    with open(RULES_PATH, 'w', encoding='utf-8') as f:
        json.dump(new_rules, f, ensure_ascii=False, indent=2)  # 保存为 JSON 格式
    
    logger.info('Applied new rules and saved to %s' % RULES_PATH)  # 记录应用日志