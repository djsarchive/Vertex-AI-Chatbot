import streamlit as st
import os
from google.cloud import storage
from google.oauth2 import service_account
import vertexai
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd
from vertexai.generative_models import GenerativeModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import tempfile
import io
import altair as alt
import json
from datetime import datetime
import re

load_dotenv()

# Configuration - Load from environment variables 
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")

class SimplePOCChatbot:
    def __init__(self, project_id, location, bucket_name=None):
        # Initialize credentials
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path:
            key_path = key_path.strip('"')
            self.credentials = service_account.Credentials.from_service_account_file(key_path)
        else:
            self.credentials = None
        
        # Initialize Vertex AI
        vertexai.init(project=project_id,location=location,credentials=self.credentials)
        
        # Initialize BigQuery client with proper credentials
        if self.credentials:
            self.bq_client = bigquery.Client(project=project_id,credentials=self.credentials)
        else:
            self.bq_client = bigquery.Client(project=project_id)
        
        # Store project info
        self.project_id = project_id
        self.location = location
        
        # Initialize Gemini model
        self.model = GenerativeModel("gemini-2.5-flash")
        
        # Initialize embeddings
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=project_id,
            location=location,
            credentials=self.credentials
        )
        
        # Initialize GCS client
        if self.credentials:
            self.storage_client = storage.Client(
                project=project_id,
                credentials=self.credentials
            )
        else:
            self.storage_client = storage.Client(project=project_id)
            
        # Initialize memory for conversation history
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Store query context and results for better continuity
        self.query_context = []
        self.last_query_result = None
        self.last_query_question = None

    def add_to_memory(self, human_message: str, ai_message: str):
        """Add messages to conversation memory"""
        self.memory.save_context(
            {"input": human_message},
            {"output": ai_message}
        )

    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""
        
        history = []
        for message in messages[-10:]:  # Last 10 messages
            if isinstance(message, HumanMessage):
                history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                history.append(f"Assistant: {message.content}")
        
        return "\n".join(history)

    def store_query_context(self, question: str, sql_query: str = None, result_df: pd.DataFrame = None, summary: str = None):
        """Store context of the current query for future reference"""
        context_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "sql_query": sql_query,
            "result_summary": summary,
            "row_count": len(result_df) if result_df is not None else 0,
            "columns": list(result_df.columns) if result_df is not None else []
        }
        
        self.query_context.append(context_entry)
        # Keep only last 5 queries for context
        if len(self.query_context) > 5:
            self.query_context.pop(0)
        
        # Store for immediate reference
        self.last_query_result = result_df
        self.last_query_question = question

    def get_query_context_string(self) -> str:
        """Get formatted query context for prompt"""
        if not self.query_context:
            return ""
        
        context_str = "Recent Query Context:\n"
        for i, ctx in enumerate(self.query_context[-3:], 1):  # Last 3 queries
            context_str += f"{i}. Question: {ctx['question']}\n"
            if ctx['sql_query']:
                context_str += f"   SQL: {ctx['sql_query'][:100]}...\n"
            context_str += f"   Result: {ctx['result_summary'][:150]}...\n"
            context_str += f"   Rows: {ctx['row_count']}, Columns: {ctx['columns']}\n\n"
        
        return context_str

    def is_data_query(self, question: str) -> bool:
        """Determine if the question is asking for data analysis or just conversation"""
        # Keywords that indicate data queries
        data_keywords = [
            'show', 'display', 'get', 'find', 'search', 'list', 'count', 'sum', 'average', 'max', 'min',
            'table', 'data', 'records', 'rows', 'columns', 'database', 'query', 'select', 'where',
            'group by', 'order by', 'filter', 'sort', 'total', 'calculate', 'analyze', 'report',
            'how many', 'what is the', 'give me', 'fetch', 'retrieve', 'export', 'chart', 'graph',
            'plot', 'visualize', 'dashboard', 'metric', 'kpi', 'trend', 'comparison', 'breakdown'
        ]
        
        # Keywords that indicate non-data queries
        non_data_keywords = [
            'hello', 'hi', 'thanks', 'thank you', 'goodbye', 'bye', 'help', 'how are you',
            'what are you', 'who are you', 'explain', 'define', 'what is', 'how to', 'why',
            'tell me about', 'describe', 'summarize', 'opinion', 'think', 'feel', 'recommend',
            'suggest', 'advice', 'tutorial', 'guide', 'example', 'sample'
        ]
        
        question_lower = question.lower()
        
        # Check for explicit non-data patterns first
        non_data_patterns = [
            r'\b(what|who|why|how) (is|are|do|does|can|should|would)\b',
            r'\b(explain|define|describe|tell me about)\b',
            r'\b(hello|hi|thanks|thank you|goodbye|bye)\b',
            r'\b(help|tutorial|guide|example)\b'
        ]
        
        for pattern in non_data_patterns:
            if re.search(pattern, question_lower):
                return False
        
        # Check for data query patterns
        data_patterns = [
            r'\b(show|display|get|find|list|count|sum|average|total)\b',
            r'\b(how many|what is the (count|sum|average|total))\b',
            r'\b(chart|graph|plot|visualize)\b',
            r'\b(table|database|query|select|where)\b',
            r'\b(report|analyze|breakdown|comparison)\b'
        ]
        
        for pattern in data_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # Check for references to previous data queries
        if any(word in question_lower for word in ['previous', 'last', 'that data', 'those results', 'above']):
            return True
            
        # If no clear pattern, check keyword counts
        data_count = sum(1 for keyword in data_keywords if keyword in question_lower)
        non_data_count = sum(1 for keyword in non_data_keywords if keyword in question_lower)
        
        if data_count > non_data_count and data_count > 0:
            return True
            
        return False

    def test_bigquery_connection(self):
        try:
            datasets = list(self.bq_client.list_datasets(max_results=1))
            return True
        except Exception as e:
            print(f"BigQuery connection test failed: {str(e)}")
            return False

    def query_bigquery(self, sql_query):
        try:

            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            self.bq_client.query(sql_query, job_config=job_config)
            
            query_job = self.bq_client.query(sql_query)
            df = query_job.result().to_dataframe()
            return df
        except Exception as e:
            print(f"Direct SQL query error: {str(e)}")
            return pd.DataFrame()

    def get_dynamic_schema_prompt(self):
        try:
            datasets = list(self.bq_client.list_datasets())
            schema_description = ""
            for dataset in datasets:
                dataset_id = dataset.dataset_id
                tables = self.bq_client.list_tables(dataset.dataset_id)
                for table in tables:
                    table_ref = f"{dataset.dataset_id}.{table.table_id}"
                    schema_description += f"- {table_ref} ("
                    table_obj = self.bq_client.get_table(table)
                    columns = [f"{schema.name} {schema.field_type}" for schema in table_obj.schema]
                    schema_description += ", ".join(columns) + ")\n"
            return f"Schema:\n{schema_description}"
        except Exception as e:
            return "Schema information could not be retrieved."

    def generate_sql_from_question(self, question: str) -> pd.DataFrame:
        schema_prompt = self.get_dynamic_schema_prompt()
        conversation_history = self.get_conversation_history()
        query_context = self.get_query_context_string()

        prompt = f"""
You are a BigQuery SQL generator with conversation memory. Your task is to convert natural language questions into valid BigQuery SQL queries, considering the conversation context.

{schema_prompt}

CONVERSATION HISTORY:
{conversation_history}

{query_context}

CONTEXT AWARENESS RULES:
1. If the user refers to "previous query", "last result", "that data", etc., use the context above
2. If asking for modifications (like "show me more columns", "filter that", "group by"), build upon previous queries
3. If asking follow-up questions about the same topic, maintain consistency
4. If it's a completely new topic, start fresh

IMPORTANT INSTRUCTIONS:
1. Return ONLY the SQL query, nothing else
2. Do not include any explanations, markdown formatting, or additional text
3. The query must start with SELECT
4. If you cannot generate a valid SQL query for the question, return: NO_SQL_POSSIBLE

Current User Question: {question}

SQL Query:"""
        
        try:
            response = self.model.generate_content(prompt)
            generated_text = response.text.strip()
            
            # Handle markdown formatting if present
            if "```sql" in generated_text:
                # Extract SQL from markdown code block
                sql_start = generated_text.find("```sql") + 6
                sql_end = generated_text.find("```", sql_start)
                if sql_end != -1:
                    generated_sql = generated_text[sql_start:sql_end].strip()
                else:
                    generated_sql = generated_text[sql_start:].strip()
            elif "```" in generated_text:
                # Extract from generic code block
                sql_start = generated_text.find("```") + 3
                sql_end = generated_text.find("```", sql_start)
                if sql_end != -1:
                    generated_sql = generated_text[sql_start:sql_end].strip()
                else:
                    generated_sql = generated_text[sql_start:].strip()
            else:
                generated_sql = generated_text
            
            # Clean up the SQL
            generated_sql = generated_sql.strip()
            
            # Check if model indicated it can't generate SQL
            if "NO_SQL_POSSIBLE" in generated_sql.upper():
                print(f"Model indicated SQL generation not possible for: {question}")
                return pd.DataFrame()
            
            # Look for SELECT statement in the text (case insensitive)
            lines = generated_sql.split('\n')
            sql_query = None
            
            for i, line in enumerate(lines):
                if line.strip().upper().startswith("SELECT"):
                    # Found SELECT, take this line and all following lines
                    sql_query = '\n'.join(lines[i:]).strip()
                    break
            
            if not sql_query:
                # Last resort: check if any line contains SELECT
                for line in lines:
                    if "SELECT" in line.upper():
                        sql_query = generated_sql
                        break
            
            if not sql_query or not sql_query.upper().strip().startswith("SELECT"):
                print(f"Generated text does not contain valid SQL: {generated_text[:200]}...")
                return pd.DataFrame()

            print(f"Generated SQL: {sql_query}")
            
            # Dry run validation
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            self.bq_client.query(sql_query, job_config=job_config)

            # Execute query
            query_job = self.bq_client.query(sql_query)
            df = query_job.result().to_dataframe()
            
            # Store the SQL query for context
            self.last_sql_query = sql_query
            
            return df
            
        except Exception as e:
            print(f"SQL Generation or Execution Error: {str(e)}")
            return pd.DataFrame()

    def summarize_sql_results(self, df: pd.DataFrame, original_question: str) -> str:
        if df.empty:
            return "No data was found for the given question."

        table_summary = df.head(10).to_markdown(index=False)
        conversation_history = self.get_conversation_history()
        query_context = self.get_query_context_string()

        prompt = f"""
You are analyzing SQL query results to provide quick, actionable insights. Keep your response concise and focused.

ORIGINAL QUESTION: {original_question}
CONVERSATION HISTORY: {conversation_history}
{query_context}
CURRENT QUERY RESULTS: {table_summary}

ANALYSIS REQUIREMENTS:
1. Answer the original question directly in 1-2 sentences
2. Highlight 2-3 key insights using bullet points
3. Mention any important patterns or trends
4. Keep the total response under 150 words

Format your response as:
**Quick Answer:** [Direct answer to the question]

**Key Insights:**
• [Insight 1]
• [Insight 2] 
• [Insight 3 if relevant]

**Bottom Line:** [One sentence summarizing what this means for the business]
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return "Could not analyze the results with context."

    def format_dataframe_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        formatted_df = df.copy()

        for col in formatted_df.columns:
            if pd.api.types.is_numeric_dtype(formatted_df[col]):
                # Handle numeric columns
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) and float(x).is_integer()
                    else f"{x:,.2f}" if pd.notna(x)
                    else x
                )
            elif pd.api.types.is_string_dtype(formatted_df[col]):
                # Handle string columns
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: str(x)[:50] + "..." if isinstance(x, str) and len(str(x)) > 50
                    else x
                )
            elif pd.api.types.is_datetime64_any_dtype(formatted_df[col]):
                # Handle datetime columns
                formatted_df[col] = formatted_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        return formatted_df

    def get_chart(self, df: pd.DataFrame):
        if df.empty:
            return None
        
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        try:
            # Time series chart
            if datetime_columns and numeric_columns:
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X(datetime_columns[0], title=datetime_columns[0]),
                    y=alt.Y(numeric_columns[0], title=numeric_columns[0])
                ).properties(width=600, height=400)
                return chart
            
            # Bar chart for categorical data
            elif non_numeric_columns and numeric_columns:
                df_limited = df.head(20) if len(df) > 20 else df
                chart = alt.Chart(df_limited).mark_bar().encode(
                    x=alt.X(non_numeric_columns[0], sort='-y', title=non_numeric_columns[0]),
                    y=alt.Y(numeric_columns[0], title=numeric_columns[0])
                ).properties(width=600, height=400)
                return chart
            
            # Scatter plot for two numeric columns
            elif len(numeric_columns) >= 2:
                df_limited = df.head(100) if len(df) > 100 else df
                chart = alt.Chart(df_limited).mark_circle(size=60).encode(
                    x=alt.X(numeric_columns[0], title=numeric_columns[0]),
                    y=alt.Y(numeric_columns[1], title=numeric_columns[1])
                ).properties(width=600, height=400)
                return chart
            
            # Simple bar chart for single numeric column
            elif len(numeric_columns) == 1:
                df_indexed = df.head(20).reset_index()
                chart = alt.Chart(df_indexed).mark_bar().encode(
                    x=alt.X('index:O', title='Index'),
                    y=alt.Y(f'{numeric_columns[0]}:Q', title=numeric_columns[0])
                ).properties(width=600, height=400)
                return chart
                
        except Exception as e:
            return None
        
        return None

    def handle_text_only_query(self, user_query: str) -> str:
        """Handle non-data queries with conversational responses"""
        conversation_history = self.get_conversation_history()
        query_context = self.get_query_context_string()
        
        prompt = f"""
You are a helpful AI assistant. The user is asking a question that doesn't require data analysis or SQL queries.

CONVERSATION HISTORY:
{conversation_history}

{query_context}

USER QUESTION: {user_query}

Please provide a helpful, conversational response. Consider the context of our previous conversation if relevant.
If this is a greeting, respond warmly. If it's a question about concepts, provide clear explanations.
If it's asking for advice or opinions, be thoughtful and helpful.

Keep your response natural and conversational, not overly formal.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return "I'd be happy to help with that, but I'm having trouble generating a response right now. Could you please try rephrasing your question?"

    def chat(self, user_query):
        response_parts = []
        chart = None
        chart_intent_keywords = ["chart", "plot", "graph", "visualize", "show", "display"]

        # First, determine if this is a data query or text-only query
        is_data_request = self.is_data_query(user_query)
        
        if is_data_request:
            # Try to get SQL results for data queries
            df_result = self.generate_sql_from_question(user_query)

            if df_result is not None and not df_result.empty:
                # Create chart if requested OR if data has numeric columns
                should_create_chart = (
                    any(kw in user_query.lower() for kw in chart_intent_keywords) or
                    len(df_result.select_dtypes(include=['number']).columns) > 0
                )
                
                if should_create_chart:
                    chart = self.get_chart(df_result)

                # Use enhanced summary with context
                summary = self.summarize_sql_results(df_result, user_query)
                
                # Store query context for future reference
                sql_query = getattr(self, 'last_sql_query', None)
                self.store_query_context(user_query, sql_query, df_result, summary)
                
                response_parts.append("**Query Results:**")
                
                # Format the DataFrame to avoid exponential notation
                formatted_df = self.format_dataframe_for_display(df_result)
                response_parts.append(formatted_df.to_markdown(index=False))
                
                response_parts.append(f"**Analysis:**\n{summary}")
                
                if chart is not None:
                    response_parts.append("**Visualization created**")
                    
                final_response = "\n\n".join(response_parts)
                
            else:
                # Data query but no results - provide explanation
                final_response = self._get_fallback_response(user_query)
        else:
            # Handle text-only queries
            final_response = self.handle_text_only_query(user_query)

        # Add to conversation memory
        self.add_to_memory(user_query, final_response)
        
        return final_response, chart

    def _get_fallback_response(self, user_query):
        """Generate a fallback response when neither SQL nor document search works"""
        conversation_history = self.get_conversation_history()
        query_context = self.get_query_context_string()
        
        try:
            fallback_prompt = f"""
The user asked: "{user_query}"

CONVERSATION HISTORY:
{conversation_history}

{query_context}

I attempted to convert this to a SQL query but was unsuccessful. Considering our conversation history and previous queries, please provide a helpful response that:

1. Acknowledges the context of our conversation
2. Explains why this query couldn't be processed as a data query
3. Suggests how they might rephrase it for data analysis
4. Offers relevant follow-up questions based on previous queries
5. Provides a general helpful response if it's not a data question

Available database schema:
{self.get_dynamic_schema_prompt()}
"""
            response = self.model.generate_content(fallback_prompt)
            return response.text.strip()
        except Exception as e:
            return f"I couldn't process your query as a data request. Based on our conversation, this might be because the question isn't related to the available data. Please try rephrasing your question or ask about specific data from the available tables."

    def clear_memory(self):
        """Clear conversation memory and query context"""
        self.memory.clear()
        self.query_context = []
        self.last_query_result = None
        self.last_query_question = None

def main():
    st.set_page_config(
        page_title="Tecnoprism AI Chatbot",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("Tecnoprism's AI powered Chatbot")
    st.caption("Powered by Tecnoprism")
    
    # Auto-initialize chatbot on first run
    if 'chatbot' not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                st.session_state.chatbot = SimplePOCChatbot(PROJECT_ID, "us-central1", GCS_BUCKET_NAME)
                st.session_state.bucket_name = GCS_BUCKET_NAME
                    
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.info("Make sure you have the proper GCP credentials and permissions")
            st.stop()
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(message["content"]["text"])
                if message["content"].get("chart"):
                    st.altair_chart(message["content"]["chart"], use_container_width=True)
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask any question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, chart = st.session_state.chatbot.chat(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display chart if generated
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                    
                    # Store response in chat history
                    message_content = {
                        "text": response,
                        "chart": chart
                    }
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": message_content
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()