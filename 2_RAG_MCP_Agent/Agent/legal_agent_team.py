import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.document import DocumentChunking

def init_session_state():
    """初始化会话状态变量"""
    st.session_state.openai_api_key = os.getenv("API_KEY", None)
    st.session_state.openai_base_url = os.getenv("BASE_URL", None)
    st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    st.session_state.qdrant_url = os.getenv("QDRANT_URL", None)
    st.session_state.vector_db = None
    st.session_state.legal_team = None
    st.session_state.knowledge_base = None
    st.session_state.processed_files = set()

COLLECTION_NAME = "legal_documents"

def init_qdrant():
    """初始化 Qdrant 向量数据库连接"""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        vector_db = Qdrant(
            collection=COLLECTION_NAME,
            api_key=st.session_state.qdrant_api_key,
            url=st.session_state.qdrant_url,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=st.session_state.openai_api_key,
                base_url=st.session_state.openai_base_url
            )
        )
        return vector_db
    except Exception as e: 
        st.error(f"初始化 Qdrant 时出错: {e}")
        return None
    
def process_document(upload_file, vector_db: Qdrant):
    if not all([st.session_state.openai_api_key, st.session_state.openai_base_url]):
        st.error("必须设置 OpenAI API 密钥和基础 URL。")
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(upload_file.read())
            tmp_file_path = tmp_file.name

        st.info("正在加载和处理文档...")
        
        knowledge = Knowledge(
            vector_db=vector_db
        )

        with st.spinner("📤 正在将文档加载到知识库中..."):
            try:
                knowledge.add_content(
                    path=tmp_file_path,
                    reader=PDFReader(
                        name="文档分块阅读器",
                        chunking_stategy=DocumentChunking(
                            chunk_size=1000,
                            overlap=200
                        )
                    )
                )
            except Exception as e:
                st.error(f"加载文档时出错: {str(e)}")
            try:
                os.unlink(tmp_file_path)
            except:
                pass

            return knowledge
    except Exception as e:
        st.error(f"文档处理错误: {str(e)}")
        raise Exception(f"处理文档时出错: {str(e)}")



def main():
    st.set_page_config(page_title="法律文件分析器", layout="wide")
    load_dotenv()
    init_session_state()

    st.title("AI 法律智能体团队 👨‍⚖️")
    with st.sidebar:
        st.header("🔑 API 配置")
        openai_key = st.text_input(
            "OpenAI API 密钥",
            type="password",
            value= st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="输入您的 OpenAI API 密钥"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
        
        openai_url = st.text_input(
            "OpenAI 基础 URL",
            type="password",
            value = st.session_state.openai_base_url if st.session_state.openai_base_url else "",
            help="输入您的 OpenAI 基础 URL"
        )
        if openai_url:
            st.session_state.openai_base_url = openai_url

        qdrant_key = st.text_input(
            "Qdrant API 密钥",
            type="password",
            value= st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else "",
            help="输入您的 Qdrant API 密钥"
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key
        
        qdrant_url = st.text_input(
            "Qdrant URL",
            type="password",
            value=st.session_state.qdrant_url if st.session_state.qdrant_url else "",
            help="输入您的 Qdrant URL"
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        
        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            try:
                if not st.session_state.vector_db:
                    st.session_state.vector_db = init_qdrant()
                    if st.session_state.vector_db:
                        st.success("Qdrant 向量数据库初始化成功！")
            except Exception as e:
                st.error(f"连接 Qdrant 失败: {e}")
        
        st.divider()

        if all([st.session_state.openai_api_key, st.session_state.openai_base_url, st.session_state.vector_db]):
            st.header("📂 知识库")
            upload_file = st.file_uploader("上传法律文件", type=['pdf'])

            if upload_file:
                if upload_file.name not in st.session_state.processed_files:
                    with st.spinner("正在处理文档..."):
                        try:
                            knowledge_base = process_document(upload_file, st.session_state.vector_db)

                            if knowledge_base:
                                st.session_state.knowledge_base = knowledge_base
                                st.session_state.processed_files.add(upload_file.name)
                                legal_researcher = Agent(
                                    name="法律研究员",
                                        role="法律研究专家",
                                        model=OpenAIChat(id="gpt-4.1"),
                                        tools=[DuckDuckGoTools()],
                                        knowledge=st.session_state.knowledge_base,
                                        search_knowledge=True,
                                        instructions=[
                                            "查找并引用相关的法律案例和先例",
                                            "提供详细的研究摘要并附上来源",
                                            "引用上传文档中的具体章节",
                                            "始终在知识库中搜索相关信息"
                                        ],
                                        markdown=True
                                )

                                contract_analyst = Agent(
                                        name="合同分析师",
                                        role="合同分析专家",
                                        model=OpenAIChat(id="gpt-4.1"),
                                        knowledge=st.session_state.knowledge_base,
                                        search_knowledge=True,
                                        instructions=[
                                            "仔细审查合同",
                                            "识别关键条款和潜在问题",
                                            "引用文档中的具体条款"
                                        ],
                                        markdown=True
                                    )

                                legal_strategist = Agent(
                                    name="法律策略师", 
                                    role="法律策略专家",
                                    model=OpenAIChat(id="gpt-4.1"),
                                    knowledge=st.session_state.knowledge_base,
                                    search_knowledge=True,
                                    instructions=[
                                        "制定全面的法律策略",
                                        "提供可行的建议",
                                        "同时考虑风险和机遇"
                                    ],
                                    markdown=True
                                )

                                st.session_state.legal_team = Team(
                                    name = "法律团队负责人",
                                    role="法律团队协调员",
                                    model=OpenAIChat(id="gpt-4.1"),
                                    members=[legal_researcher, contract_analyst, legal_strategist],
                                    knowledge=st.session_state.knowledge_base,
                                    search_knowledge=True,
                                    instructions=[
                                    "协调团队成员之间的分析",
                                            "提供全面的回应",
                                            "确保所有建议都有适当的来源",
                                            "引用上传文档的具体部分",
                                            "在委派任务之前始终搜索知识库" 
                                    ],
                                    show_members_responses=True,
                                    markdown=True
                                )

                                st.success("✅ 文档处理完毕，团队已初始化！")
                        except Exception as e:
                            st.error(f"处理文档时出错: {str(e)}")
                else:
                    st.success("✅ 文档已处理，团队已准备就绪！")
            st.divider()
            st.header("🔍 分析选项")
            analysis_type = st.selectbox(
                "选择分析类型",
                [
                    "合同审查",
                    "法律研究",
                    "风险评估",
                    "合规性检查",
                    "自定义查询"
                ]
            )
        else:
            st.warning("请配置所有 API 凭据以继续")

    if not all([st.session_state.openai_api_key, st.session_state.vector_db]):
        st.info("👈 请在侧边栏配置您的 API 凭据以开始")
    elif not upload_file:
        st.info("👈 请上传法律文件以开始分析")
    elif st.session_state.legal_team:
        # 为分析类型图标创建一个字典
        analysis_icons = {
            "合同审查": "📑",
            "法律研究": "🔍",
            "风险评估": "⚠️",
            "合规性检查": "✅",
            "自定义查询": "💭"
        }

        st.header(f"{analysis_icons[analysis_type]} {analysis_type} 分析")

        analysis_configs = {
            "合同审查": {
                "query": "审查此合同，并识别关键条款、义务和潜在问题。",
                "agents": ["合同分析师"],
                "description": "详细的合同分析，重点关注条款和义务"
            },
            "法律研究": {
                "query": "研究与此文件相关的案例和先例。",
                "agents": ["法律研究员"],
                "description": "研究相关的法律案例和先例"
            },
            "风险评估": {
                "query": "分析此文件中的潜在法律风险和责任。",
                "agents": ["合同分析师", "法律策略师"],
                "description": "综合风险分析和战略评估"
            },
            "合规性检查": {
                "query": "检查此文件的监管合规性问题。",
                "agents": ["法律研究员", "合同分析师", "法律策略师"],
                "description": "全面的合规性分析"
            },
            "自定义查询": {
                "query": None,
                "agents": ["法律研究员", "合同分析师", "法律策略师"],
                "description": "使用所有可用的智能体进行自定义分析"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 当前活动的法律 AI 智能体: {', '.join(analysis_configs[analysis_type]['agents'])}")  #dictionary!!

        # 用此代码替换现有的 user_query 部分：
        if analysis_type == "自定义查询":
            user_query = st.text_area(
                "输入您的具体查询：",
                help="添加您想分析的任何具体问题或要点"
            )
        else:
            user_query = None  # 对于非自定义查询，设置为空


        if st.button("分析"):
            if analysis_type == "自定义查询" and not user_query:
                st.warning("请输入查询内容")
            else:
                with st.spinner("正在分析文档..."):
                    try:
                        # 确保设置了 OpenAI API 密钥
                        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
                        os.environ["OPENAI_BASE_URL"] = st.session_state.openai_base_url

                        # 结合预定义查询和用户查询
                        if analysis_type != "自定义查询":
                            combined_query = f"""
                            使用上传的文档作为参考：
                            
                            主要分析任务: {analysis_configs[analysis_type]['query']}
                            重点领域: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            请搜索知识库并提供文档中的具体参考。
                            """
                        else:
                            combined_query = f"""
                            使用上传的文档作为参考：
                            
                            {user_query}
                            
                            请搜索知识库并提供文档中的具体参考。
                            重点领域: {', '.join(analysis_configs[analysis_type]['agents'])}
                            """

                        response = st.session_state.legal_team.run(combined_query)
                        
                        # 在选项卡中显示结果
                        tabs = st.tabs(["分析", "要点", "建议"])
                        
                        with tabs[0]:
                            st.markdown("### 详细分析")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### 要点")
                            key_points_response = st.session_state.legal_team.run(
                                f"""基于之前的分析：    
                                {response.content}
                                
                                请用项目符号总结要点。
                                重点关注来自以下方面的见解： {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### 建议")
                            recommendations_response = st.session_state.legal_team.run(
                                f"""基于之前的分析：
                                {response.content}
                                
                                根据分析，您的主要建议是什么，最佳行动方案是什么？
                                提供来自以下方面的具体建议： {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"分析过程中出错: {str(e)}")
    else:
        st.info("请上传法律文件以开始分析")

if __name__=="__main__":
    main()




