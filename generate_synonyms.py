import ollama
import regex as re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SynonymGenerator:
    """
    A class to generate and refine synonyms for Arabic words using the Ollama Engine.
    
    Attributes:
        ollama (ollama): An instance of the Ollama engine used to generate synonyms and embeddings.
    """

    def __init__(self):
        """Initializes the SynonymGenerator with an Ollama engine instance."""
        self.ollama = ollama

    def generate_synonyms(self, word: str, model: str, num_synonyms: int):
        """
        Generates a list of synonyms for a given word using a specified model.
        
        Args:
            word (str): The word for which synonyms are to be generated.
            model (str): The model to be used for generating synonyms.
            num_synonyms (int): The number of synonyms to generate.
        
        Returns:
            str: A raw string response containing synonyms in a list format.
        """
        # Construct the prompt to be sent to the Ollama engine
        prompt = f"generate {num_synonyms} synonyms for {word} in Arabic in list format without providing the meaning. for example: [synonym1, synonym2, synonym3] I need the results as list format only."
        system_prompt = "You are a python tool that generates synonyms for Arabic words. You need to generate synonyms for a given word in Arabic without providing the meaning. You need to generate a list of synonyms in list format. For example: [synonym1, synonym2, synonym3]. another example: [سعيد, مبتهج, فرحان] I need the only in this format."
        response = self.ollama.generate(model=model, prompt=prompt, system=system_prompt)
        # Extract the content from the Ollama response
        synonyms = response['response']
        print("Response:", synonyms)

        return synonyms

    def best_synonyms(self, word: str, model: str, embedding_model:str, num_synonyms: int , similarity_threshold: float):
        """
        Filters the generated synonyms based on a cosine similarity threshold.
        
        Args:
            word (str): The original word for which synonyms are being refined.
            model (str): The model used for generating synonyms and embeddings.
            num_synonyms (int): The number of synonyms to initially generate.
            similarity_threshold (float): The threshold for cosine similarity above which synonyms are accepted.
        
        Returns:
            list: A list of synonyms that meet the similarity threshold.
        """
        # Generate initial synonyms
        synonyms = self.generate_synonyms(word, model, num_synonyms)

        # Use regex to find the list of synonyms from the response
        synonyms_list = re.findall(r'\[(.*?)\]', synonyms)
        if not synonyms_list:
            return []

        # Split the list string into actual list items
        synonyms_list = synonyms_list[0].split(',')
        synonyms_list = [synonym.strip() for synonym in synonyms_list]

        # Remove the original word from the list of synonyms, if present
        if word in synonyms_list:
            synonyms_list.remove(word)

        # Generate embeddings for the original word and each synonym
        original_word_embedding = self.ollama.embeddings(model=embedding_model, prompt=word)
        synonyms_embed_list = [self.ollama.embeddings(model=embedding_model, prompt=synonym) for synonym in synonyms_list]

        # Reshape the embeddings to be used in cosine similarity calculation
        original_word_embedding_reshaped = np.array(original_word_embedding['embedding']).reshape(1, -1)
        synonyms_embed_list_reshaped  = []
        for synonym_embedding in synonyms_embed_list:
            synonym_embedding = np.array(synonym_embedding['embedding']).reshape(1, -1)
            synonyms_embed_list_reshaped.append(synonym_embedding)

        # Calculate cosine similarity and filter synonyms
        similarity_scores = []
        for synonym_embedding in synonyms_embed_list_reshaped:
            similarity_score = cosine_similarity(original_word_embedding_reshaped, synonym_embedding)[0][0]
            print("Similarity score:", similarity_score)
            similarity_scores.append(similarity_score)
        best_synonyms = [synonyms_list[i] for i, score in enumerate(similarity_scores) if score >= similarity_threshold]

        return best_synonyms