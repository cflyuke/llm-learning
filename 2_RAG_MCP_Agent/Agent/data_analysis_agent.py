import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import operator
import os
import joblib
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY", "")
os.environ["OPENAI_BASE_URL"] = os.getenv("BASE_URL", "")

@tool
def data_statistics(path: str) -> str:
    """加载CSV文件并返回基本统计信息和数据内容。
    Args:
        path: str : CSV文件的路径
    """
    df = pd.read_csv(path)
    stats = df.describe().to_string()
    head = df.head().to_string()
    return f"数据预览:\n{head}\n\n数据统计信息:\n{stats}\n"

@tool
def plot_correlation_heatmap(path:str, output_path: str) -> str:
    """生成并保存CSV文件的相关性热力图。
    Args:
        path: str : CSV文件的路径
        output_path: str : 热力图保存路径
    """
    df = pd.read_csv(path)
    numerical_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Relative Heatmap')
    plt.savefig(output_path)
    plt.close()
    return f"相关性热力图已保存至 {output_path}"


def preprocess_titanic_data(df: pd.DataFrame) -> pd.DataFrame:
    """预处理Titanic数据集。
    参数:
        df: pd.DataFrame : 输入的数据帧
    """
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return df[features]

@tool
def train_model(path: str, output_model_path: str) -> str:
    """在Titanic数据集上训练RandomForest模型并保存。
    Args:
        path: str : CSV文件的路径
        output_model_path: str : 训练好的模型保存路径
    """
    df = pd.read_csv(path)
    X = preprocess_titanic_data(df)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 模型训练
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, output_model_path)
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    sorted_features = dict(sorted(feature_importance.items(), key=operator.itemgetter(1), reverse=True))
    
    return f"""模型已训练并保存到 {output_model_path}
测试准确率: {accuracy:.2f}
特征重要性:
{sorted_features}"""

@tool
def predict_survival(model_path: str, passenger_data: dict) -> str:
    """预测乘客的生存概率。
    参数:
        model_path: str : 保存的模型路径
        passenger_data: dict : 乘客信息字典
    """
    # 加载模型
    model = joblib.load(model_path)
    df = pd.DataFrame([passenger_data])
    # 预处理数据
    X = preprocess_titanic_data(df)
    probability = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    
    return f"""生存预测结果: {"存活" if prediction == 1 else "未存活"}
生存概率: {probability[1]:.2f}"""


class ChatBot:
    def __init__(self):
        self.tools = [data_statistics, plot_correlation_heatmap, train_model, predict_survival]
        self.name2tool = {tool.name: tool for tool in self.tools}
        
        # 初始化LLM
        self.llm_model = init_chat_model(model="gpt-4.1", temperature=0.5).bind_tools(self.tools)
        
        # 初始化对话状态
        self.conversation_history = []
        self.system_message = SystemMessage(content="你是一个数据分析助手，能够帮用户做数据分析。你可以进行数据统计分析、绘制相关性热力图、训练模型和预测。请保持对话的连续性，记住之前的上下文。")
        
        # 构建Agent
        self.agent = self._build_agent()
    
    def _build_agent(self):
        def llm_node(state: MessagesState):
            # 合并历史记录和当前消息
            all_messages = [self.system_message] + self.conversation_history + state["messages"]
            response = self.llm_model.invoke(all_messages)
            return {"messages": [response]}
        
        def tool_node(state: MessagesState):
            result = []
            for tool_call in state["messages"][-1].tool_calls:
                tool = self.name2tool[tool_call["name"]]
                tool_result = tool.invoke(tool_call["args"])
                result.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                )
            return {"messages": result}
        
        def should_continue(state: MessagesState):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tool_node"
            return END
        
        # 构建工作流程图
        agent_builder = StateGraph(MessagesState)
        agent_builder.add_node("llm_node", llm_node)
        agent_builder.add_node("tool_node", tool_node)
        agent_builder.add_edge(START, "llm_node")
        agent_builder.add_edge("tool_node", "llm_node")
        agent_builder.add_conditional_edges("llm_node", should_continue, ["tool_node", END])
        
        return agent_builder.compile()
    
    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复
        
        参数:
            user_input: str : 用户输入的消息
        返回:
            str : 助手的回复
        """
        # 创建用户消息
        user_message = HumanMessage(content=user_input)
        
        # 运行agent
        result = self.agent.invoke({"messages": [user_message]})
        
        # 更新对话历史
        self.conversation_history.extend([user_message] + result["messages"])
        
        # 返回最后一条消息的内容
        return result["messages"][-1].content
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

if __name__ == "__main__":
    chatbot = ChatBot()
    print("你好！我是你的数据分析助手 🤖 (输入 'exit' 退出, 'clear' 清空历史): ")
    
    while True:
        user_input = input("你：")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_history()
            print("对话历史已清空")
            continue
        response = chatbot.chat(user_input)
        print(f"🤖：{response}")








