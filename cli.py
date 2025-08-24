import argparse
from rag.rag_pipeline import ingest_pdfs, answer_question

def main():
    parser = argparse.ArgumentParser(description="PDF RAG CLI")
    sub = parser.add_subparsers(dest="cmd")

    ing = sub.add_parser("ingest", help="Ingest one or more PDFs")
    ing.add_argument("paths", nargs="+", help="Paths to PDF files")

    ask = sub.add_parser("ask", help="Ask a question against the index")
    ask.add_argument("query", help="Your question")
    ask.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")

    args = parser.parse_args()

    if args.cmd == "ingest":
        res = ingest_pdfs(args.paths)
        print(f"Ingested chunks: {res['chunks_added']}")
    elif args.cmd == "ask":
        res = answer_question(args.query, top_k=args.top_k)
        print("\nAnswer:\n")
        print(res["answer"])
        print("\nSources:")
        for s in res["sources"]:
            print(f"- {s['filename']} page {s['page']} score {s['score']:.3f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
