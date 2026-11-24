import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

# Supabase connection string format:
# postgresql://postgres:[YOUR-DATABASE-PASSWORD]@db.srofdjrilutpxgjznxsl.supabase.co:5432/postgres
#
# Get these from: Supabase Dashboard -> Project Settings -> Database

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

# For demo, use local if Supabase not configured
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)


def connect_to_supabase():
    """Connect to Supabase pgvector"""

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        check_embedding_ctx_length=False,
    )

    # Note: Supabase has pgvector pre-installed
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    return vectorstore


def verify_connection(vectorstore):
    """Verify the connection works"""
    from langchain_core.documents import Document

    # Add a test document
    test_doc = Document(
        page_content="This is a test document to verify Supabase",
        metadata={"test": True},
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"✅ Added test document: {ids[0]}")

        # Search for it
        results = vectorstore.similarity_search("test document")
        if results:
            print(f"✅ Search workds: {results[0].page_content}")

        # Clean Up
        # vectorstore.delete(ids)
        # print("✅ Cleanup complete")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        host = SUPABASE_URL.rsplit("@", 1)[1].split(":")[0].split("/")[0]
        print("\n🔗 Connecting to Supabase...")
        print(f"  Host: {host}")
    else:
        print("\n⚠️ SUPABASE_DATABASE_URL not set")
        print("  Using local PostgreSQL instead")

    vectorstore = connect_to_supabase()
    print("\n✅ Connected!")

    print("\n🧪 Running verification...")
    success = verify_connection(vectorstore)

    if success:
        print("\n✅ Supabase pgvector is ready for production!")
    else:
        print("\n❌ Verification failed - check connection string")


if __name__ == "__main__":
    main()
