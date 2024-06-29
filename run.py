import argparse
from generate_synonyms import SynonymGenerator

def main(args):
    synonym_generator = SynonymGenerator()
    best_synonyms = synonym_generator.best_synonyms(args.word, args.model, args.embedding_model, args.num_synonyms, args.similarity_threshold)
    print(best_synonyms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synonyms for a given word')
    parser.add_argument('--word', type=str, help='Word to generate synonyms for')
    parser.add_argument('--model', type=str, default="aya", help='Model to use for generating synonyms')
    parser.add_argument('--embedding_model', type=str, default="llama3", help='Model to use for generating embeddings')
    parser.add_argument('--num_synonyms', type=int, default=10, help='Number of synonyms to generate')
    parser.add_argument('--similarity_threshold', type=float, default=0.8, help='Cosine similarity threshold for selecting best synonyms')

    args = parser.parse_args()
    main(args)
