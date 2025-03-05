import os
import pickle
from pathlib import Path

from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

# -----------------------------------------------------------------------------
# Custom Response Parser for Streamlit
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        """Display dataframe using Streamlit and save a placeholder message."""
        st.dataframe(result["value"])
        # Simpan placeholder ke session state agar tidak menghasilkan None
        st.session_state.messages.append({"role": "assistant", "content": "[Displayed DataFrame]"})
        return

    def format_plot(self, result):
        """Display plot image using Streamlit and save a placeholder message."""
        st.image(result["value"])
        # Simpan placeholder ke session state agar tidak menghasilkan None
        st.session_state.messages.append({"role": "assistant", "content": "[Displayed Plot]"})
        return

    def format_other(self, result):
        """Display other types of results as text and simpan ke session state."""
        st.write(str(result["value"]))
        st.session_state.messages.append({"role": "assistant", "content": str(result["value"])})
        return

# -----------------------------------------------------------------------------
# Validate Database Connection and Load Tables
# -----------------------------------------------------------------------------
def validate_and_connect_database(credentials):
    try:
        # Extract credentials
        db_user = credentials["DB_USER"]
        db_password = credentials["DB_PASSWORD"]
        db_host = credentials["DB_HOST"]
        db_port = credentials["DB_PORT"]
        db_name = credentials["DB_NAME"]
        groq_api_key = credentials["GROQ_API_KEY"]

        # Encode password for special characters
        encoded_password = db_password.replace('@', '%40')

        # Create database engine
        engine = create_engine(
            f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        )

        with engine.connect() as connection:
            # Initialize LLM menggunakan ChatGroq
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

            # Inspect database untuk mendapatkan tabel dan view dari schema public
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables_views = tables + views

            sdf_list = []
            table_info = {}

            for table in all_tables_views:
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)
                    # Buat SmartDataframe dengan LLM dan custom response parser
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm, "response_parser": StreamlitResponse(st)}
                    sdf_list.append(sdf)
                    # Simpan metadata tabel
                    table_info[table] = {
                        "columns": list(df.columns),
                        "row_count": len(df)
                    }
                except Exception as e:
                    st.warning(f"Failed to load data from public.{table}: {e}")

            # Buat SmartDatalake dari list SmartDataframe
            datalake = SmartDatalake(sdf_list, config={"llm": llm, "response_parser": StreamlitResponse})
            return datalake, table_info, engine

    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

# -----------------------------------------------------------------------------
# Cache database tables using pickle
# -----------------------------------------------------------------------------
def load_database_cache(credentials, cache_path="db_cache.pkl"):
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                datalake, table_info = pickle.load(f)
            return datalake, table_info
        except Exception as e:
            st.warning(f"Failed to load cache: {e}. Reloading data from database.")

    datalake, table_info, engine = validate_and_connect_database(credentials)
    if datalake is not None and table_info is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((datalake, table_info), f)
        except Exception as e:
            st.warning(f"Failed to save cache: {e}")
    return datalake, table_info

# -----------------------------------------------------------------------------
# Main function for Streamlit app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    st.title("🔍 Smart Database Explorer")

    # Inisialisasi session state jika belum ada
    if "database_loaded" not in st.session_state:
        st.session_state.database_loaded = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar untuk database credentials
    with st.sidebar:
        st.header("🔐 Database Credentials")
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        connect_button = st.button("Connect to Database")

    # Attempt koneksi database jika tombol ditekan dan semua credentials terisi
    if connect_button and all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
        credentials = {
            "DB_USER": db_user,
            "DB_PASSWORD": db_password,
            "DB_HOST": db_host,
            "DB_PORT": db_port,
            "DB_NAME": db_name,
            "GROQ_API_KEY": groq_api_key
        }
        with st.spinner("Connecting to the database and loading tables..."):
            datalake, table_info = load_database_cache(credentials)

        if datalake and table_info:
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.database_loaded = True

    # Jika database sudah dimuat, tampilkan informasi tabel dan chat
    if st.session_state.database_loaded:
        st.header("💬 Database Chat")

        st.subheader("📊 Loaded Tables")
        for table, info in st.session_state.table_info.items():
            with st.expander(table):
                st.write(f"Columns: {', '.join(info['columns'])}")
                st.write(f"Row Count: {info['row_count']}")

        # Tampilkan riwayat percakapan
        for message in st.session_state.messages:
            avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input untuk query pengguna
        prompt = st.chat_input("Ask a question about your data")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            with st.spinner("Generating response..."):
                try:
                    # Respons akan di-render langsung oleh StreamlitResponse
                    answer = st.session_state.datalake.chat(prompt)
                    # Jika answer tidak None, simpan ke chat history
                    if answer is not None:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error processing chat: {e}")

if __name__ == "__main__":
    main()
