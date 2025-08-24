import argparse
import os
from rag.rag_pipeline import ingest_pdfs, answer_question

def main():
    parser = argparse.ArgumentParser(description="PDF RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser('ingest', help='Ingest PDF(s)')
    ingest_parser.add_argument('pdfs', nargs='+', help='PDF file paths')

    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--top_k', type=int, default=3, help='Number of chunks to retrieve')

    args = parser.parse_args()

    if args.command == "ingest":
        res = ingest_pdfs(args.pdfs)
        print(f"Successfully ingested: {len(res['chunks_added'])} chunks")
    elif args.command == "ask":
        res = answer_question(args.question, top_k=args.top_k)
        print("\nAnswer:\n" + res["answer"])
        print("\nSources:")
        for s in res["sources"]:
            print(s)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
