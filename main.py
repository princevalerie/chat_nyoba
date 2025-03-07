import os
import pickle
from pathlib import Path
import numpy as np
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

# Fungsi untuk mencatat error ke session state
def log_error(message):
    st.session_state.error_logs.append(message)
    st.warning(message)

# -----------------------------------------------------------------------------
# Custom Response Parser for Streamlit
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        """Tampilkan dataframe dan simpan pesan placeholder ke cache."""
        st.dataframe(result["value"])
        st.session_state.answer_cache.append("[Displayed DataFrame]")
        return

    def format_plot(self, result):
        """Tampilkan plot dan simpan pesan placeholder ke cache."""
        st.image(result["value"])
        st.session_state.answer_cache.append("[Displayed Plot]")
        return

    def format_other(self, result):
        """Tampilkan hasil lain sebagai teks dan simpan ke cache."""
        st.write(str(result["value"]))
        st.session_state.answer_cache.append(str(result["value"]))
        return

# -----------------------------------------------------------------------------
# Validate Database Connection and Load Tables
# -----------------------------------------------------------------------------
def validate_and_connect_database(credentials):
    try:
        db_user = credentials["DB_USER"]
        db_password = credentials["DB_PASSWORD"]
        db_host = credentials["DB_HOST"]
        db_port = credentials["DB_PORT"]
        db_name = credentials["DB_NAME"]
        groq_api_key = credentials["GROQ_API_KEY"]

        encoded_password = db_password.replace('@', '%40')
        engine = create_engine(
            f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        )

        with engine.connect() as connection:
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)
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
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm, "response_parser": StreamlitResponse(st)}
                    sdf_list.append(sdf)
                    table_info[table] = {
                        "columns": list(df.columns),
                        "row_count": len(df)
                    }
                except Exception as e:
                    err_msg = f"Failed to load data from public.{table}: {e}"
                    log_error(err_msg)
            datalake = SmartDatalake(sdf_list, config={"llm": llm, "response_parser": StreamlitResponse})
            return datalake, table_info, engine

    except Exception as e:
        err_msg = f"Database connection error: {e}"
        log_error(err_msg)
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
            log_error(f"Failed to load cache: {e}. Reloading data from database.")
    datalake, table_info, engine = validate_and_connect_database(credentials)
    if datalake is not None and table_info is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((datalake, table_info), f)
        except Exception as e:
            log_error(f"Failed to save cache: {e}")
    return datalake, table_info

# -----------------------------------------------------------------------------
# Main function for Streamlit app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Database Explorer", layout="wide")
    st.title("🔍 AI Database Explorer")

    # Inisialisasi session state jika belum ada
    if "database_loaded" not in st.session_state:
        st.session_state.database_loaded = False
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = []  # Container sementara untuk cache answer
    if "error_logs" not in st.session_state:
        st.session_state.error_logs = []  # Container untuk error log

    # Sidebar: Database credentials, status, dan error log
    with st.sidebar:
        st.header("🔐 Database Credentials")
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        connect_button = st.button("Connect to Database")

        if st.session_state.get("database_loaded", False):
            st.subheader("📊 Loaded Tables")
            for table, info in st.session_state.table_info.items():
                with st.expander(table):
                    st.write(f"Columns: {', '.join(info['columns'])}")
                    st.write(f"Row Count: {info['row_count']}")

        if st.session_state.error_logs:
            st.subheader("⚠️ Error Log")
            for err in st.session_state.error_logs:
                st.error(err)

    # Attempt koneksi database
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
            # Menghapus error log setelah koneksi berhasil (opsional)
            st.session_state.error_logs.clear()
            st.success("Koneksi dan loading tabel selesai.")

    # Konten utama: Input query dan output
    if st.session_state.database_loaded:
        st.header("💬 Query Data dengan Natural Language")
        with st.form(key="query_form"):
            prompt = st.text_input("Masukkan query Anda:")
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.session_state.answer_cache.clear()  # Refresh output
                with st.spinner("Generating output..."):
                    try:
                        answer = st.session_state.datalake.chat(prompt)
                        st.session_state.answer_cache.append(answer)
                    except Exception as e:
                        st.error(f"Error processing query: {e}")

        if st.session_state.answer_cache:
            st.subheader("Output")
            for ans in st.session_state.answer_cache:
                st.write(ans)

if __name__ == "__main__":
    main()
