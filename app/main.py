import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    st.title("Mail Generator for job applications")
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            docs = loader.load()

            if not docs:
                st.error("Could not load content from the URL. The page may be JavaScript-rendered or inaccessible.")
                return

            MAX_CHARS = 5000  # Limit content size to avoid LLM overflow
            raw_data = docs[0].page_content[:MAX_CHARS]

            if not raw_data.strip():
                st.error("The loaded page has no readable content.")
                return

            st.info(f"📄 Loaded content preview:\n\n{raw_data[:300]}...")

            page_data = clean_text(raw_data)

            # Optional chunking to ensure safety
            splitter = CharacterTextSplitter(separator="\n\n", chunk_size=3000, chunk_overlap=200)
            chunks = splitter.split_text(page_data)

            jobs = []
            for chunk in chunks:
                try:
                    extracted = llm.extract_jobs(chunk)
                    if extracted:
                        jobs.extend(extracted)
                        break  # stop at first successful result
                except Exception:
                    continue

            if not jobs:
                st.warning(" No job descriptions could be extracted from this URL.")
                return

            portfolio.load_portfolio()

            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')

        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
