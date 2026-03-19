# NLP Reference

## Table of Contents
1. [Text Preprocessing](#text-preprocessing)
2. [Word Embeddings](#word-embeddings)
3. [Sentence Transformers](#sentence-transformers)
4. [Text Classification](#text-classification)
5. [Named Entity Recognition](#named-entity-recognition)
6. [Question Answering](#question-answering)
7. [Text Generation](#text-generation)
8. [RAG Systems](#rag-systems)

---

## Text Preprocessing

### Modern Text Cleaning

```python
import re
import unicodedata
from typing import Callable

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_special_chars: bool = False,
    remove_extra_spaces: bool = True,
) -> str:
    """Clean text with configurable options."""
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    if remove_emails:
        text = re.sub(r"\S+@\S+", "", text)
    
    if remove_special_chars:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    if lowercase:
        text = text.lower()
    
    if remove_extra_spaces:
        text = " ".join(text.split())
    
    return text.strip()


def create_text_pipeline(*funcs: Callable[[str], str]) -> Callable[[str], str]:
    """Compose multiple text processing functions."""
    def pipeline(text: str) -> str:
        for func in funcs:
            text = func(text)
        return text
    return pipeline

# Usage
pipeline = create_text_pipeline(
    lambda x: clean_text(x, lowercase=True),
    lambda x: x.replace("\n", " "),
)
```

### Tokenization Best Practices

```python
from transformers import AutoTokenizer

def tokenize_for_classification(
    texts: list[str],
    tokenizer,
    max_length: int = 512,
) -> dict:
    """Tokenize texts for classification tasks."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def tokenize_long_document(
    text: str,
    tokenizer,
    max_length: int = 512,
    stride: int = 128,
) -> list[dict]:
    """Tokenize long document with sliding window."""
    return tokenizer(
        text,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )


def smart_truncation(
    text: str,
    tokenizer,
    max_length: int = 512,
    truncation_strategy: str = "longest_first",  # or "only_first", "only_second"
) -> str:
    """Truncate text intelligently keeping important parts."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_length - 2:  # Account for [CLS] and [SEP]
        return text
    
    # Keep first and last parts
    keep_tokens = max_length - 2
    first_half = keep_tokens // 2
    second_half = keep_tokens - first_half
    
    truncated_tokens = tokens[:first_half] + tokens[-second_half:]
    return tokenizer.decode(truncated_tokens)
```

---

## Word Embeddings

### Word2Vec / FastText

```python
from gensim.models import Word2Vec, FastText
import numpy as np

def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 10,
) -> Word2Vec:
    """Train Word2Vec model."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=-1,
        sg=1,  # Skip-gram (better for rare words)
        epochs=epochs,
    )
    return model


def get_document_embedding(
    tokens: list[str],
    model: Word2Vec,
    strategy: str = "mean",
) -> np.ndarray:
    """Get document embedding from word embeddings."""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if not vectors:
        return np.zeros(model.vector_size)
    
    vectors = np.array(vectors)
    
    if strategy == "mean":
        return vectors.mean(axis=0)
    elif strategy == "max":
        return vectors.max(axis=0)
    elif strategy == "sum":
        return vectors.sum(axis=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

### TF-IDF Weighted Embeddings

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tfidf_weighted_embeddings(
    documents: list[str],
    word_vectors: dict[str, np.ndarray],
    vector_size: int = 300,
) -> np.ndarray:
    """Create TF-IDF weighted document embeddings."""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    feature_names = tfidf.get_feature_names_out()
    
    embeddings = []
    for doc_idx in range(len(documents)):
        doc_vector = np.zeros(vector_size)
        total_weight = 0
        
        for word_idx, weight in zip(
            tfidf_matrix[doc_idx].indices,
            tfidf_matrix[doc_idx].data,
        ):
            word = feature_names[word_idx]
            if word in word_vectors:
                doc_vector += weight * word_vectors[word]
                total_weight += weight
        
        if total_weight > 0:
            doc_vector /= total_weight
        
        embeddings.append(doc_vector)
    
    return np.array(embeddings)
```

---

## Sentence Transformers

### Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Any

class SemanticSearch:
    """Semantic search using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
    
    def index(self, documents: list[str], batch_size: int = 32):
        """Index documents for search."""
        self.corpus = documents
        self.corpus_embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Cosine similarity (dot product for normalized vectors)
        scores = np.dot(self.corpus_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.corpus[idx],
                "score": float(scores[idx]),
                "index": int(idx),
            })
        
        return results


# With FAISS for large-scale
import faiss

class FAISSSearch:
    """Scalable semantic search with FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = None
    
    def build_index(
        self,
        documents: list[str],
        batch_size: int = 32,
        use_gpu: bool = False,
    ):
        """Build FAISS index."""
        self.corpus = documents
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        
        dimension = embeddings.shape[1]
        
        # IVF index for large datasets
        if len(documents) > 10000:
            nlist = int(np.sqrt(len(documents)))
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(embeddings)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        return [
            {"text": self.corpus[idx], "score": float(score), "index": int(idx)}
            for score, idx in zip(scores[0], indices[0])
        ]
```

---

## Text Classification

### Fine-tuning BERT

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def prepare_classification_dataset(
    texts: list[str],
    labels: list[int],
    tokenizer,
    max_length: int = 256,
) -> Dataset:
    """Prepare dataset for classification."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


# Training
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Zero-shot Classification

```python
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,  # GPU
)

def classify_zero_shot(
    text: str,
    candidate_labels: list[str],
    multi_label: bool = False,
) -> dict:
    """Zero-shot classification without training."""
    result = classifier(
        text,
        candidate_labels,
        multi_label=multi_label,
    )
    return {
        "labels": result["labels"],
        "scores": result["scores"],
        "best": result["labels"][0],
    }
```

---

## Named Entity Recognition

### SpaCy NER

```python
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_trf")  # Transformer-based

def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text."""
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    
    return entities


def extract_entity_relations(text: str) -> list[dict]:
    """Extract entities with their relations."""
    doc = nlp(text)
    
    relations = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "dobj", "pobj"):
                head = token.head
                relations.append({
                    "subject": token.text,
                    "relation": head.text,
                    "object": head.head.text if head.head else None,
                })
    
    return relations
```

### HuggingFace NER

```python
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=0,
)

def extract_entities_bert(text: str) -> list[dict]:
    """Extract entities using BERT NER."""
    entities = ner_pipeline(text)
    
    # Merge adjacent entities of same type
    merged = []
    for entity in entities:
        if merged and entity["entity_group"] == merged[-1]["entity_group"]:
            if entity["start"] - merged[-1]["end"] <= 1:
                merged[-1]["word"] += " " + entity["word"]
                merged[-1]["end"] = entity["end"]
                merged[-1]["score"] = (merged[-1]["score"] + entity["score"]) / 2
                continue
        merged.append(entity)
    
    return merged
```

---

## Question Answering

### Extractive QA

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0,
)

def answer_question(
    question: str,
    context: str,
    max_answer_length: int = 100,
) -> dict:
    """Extract answer from context."""
    result = qa_pipeline(
        question=question,
        context=context,
        max_answer_len=max_answer_length,
    )
    
    return {
        "answer": result["answer"],
        "score": result["score"],
        "start": result["start"],
        "end": result["end"],
    }


def answer_from_multiple_contexts(
    question: str,
    contexts: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Find answers from multiple contexts."""
    answers = []
    
    for i, context in enumerate(contexts):
        result = qa_pipeline(question=question, context=context)
        answers.append({
            "answer": result["answer"],
            "score": result["score"],
            "context_index": i,
        })
    
    # Sort by score
    answers.sort(key=lambda x: x["score"], reverse=True)
    return answers[:top_k]
```

---

## Text Generation

### Prompt Templates

```python
from string import Template

class PromptTemplate:
    """Reusable prompt templates."""
    
    SUMMARIZATION = Template(
        "Summarize the following text in $length sentences:\n\n$text\n\nSummary:"
    )
    
    CLASSIFICATION = Template(
        "Classify the following text into one of these categories: $categories\n\n"
        "Text: $text\n\nCategory:"
    )
    
    QA = Template(
        "Answer the question based on the context.\n\n"
        "Context: $context\n\n"
        "Question: $question\n\n"
        "Answer:"
    )
    
    @classmethod
    def format(cls, template_name: str, **kwargs) -> str:
        template = getattr(cls, template_name.upper())
        return template.safe_substitute(**kwargs)


# Usage
prompt = PromptTemplate.format(
    "summarization",
    length="3",
    text="Long text here...",
)
```

### Generation with Constraints

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_with_constraints(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop_sequences: list[str] | None = None,
) -> str:
    """Generate text with various constraints."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create stopping criteria
    stop_ids = []
    if stop_sequences:
        for seq in stop_sequences:
            stop_ids.extend(tokenizer.encode(seq, add_special_tokens=False))
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_ids if stop_ids else tokenizer.eos_token_id,
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):]
```

---

## RAG Systems

### Complete RAG Pipeline

```python
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from typing import Any

class RAGSystem:
    """Retrieval-Augmented Generation system."""
    
    def __init__(
        self,
        retriever_model: str = "all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-base",
    ):
        self.retriever = SentenceTransformer(retriever_model)
        self.generator = pipeline("text2text-generation", model=generator_model)
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: list[str], batch_size: int = 32):
        """Add documents to the knowledge base."""
        self.documents.extend(documents)
        new_embeddings = self.retriever.encode(
            documents,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents."""
        query_embedding = self.retriever.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            {"text": self.documents[i], "score": float(scores[i])}
            for i in top_indices
        ]
    
    def generate(
        self,
        query: str,
        context: list[str],
        max_length: int = 256,
    ) -> str:
        """Generate answer from retrieved context."""
        context_str = "\n\n".join(context)
        
        prompt = (
            f"Answer the question based on the following context:\n\n"
            f"Context: {context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        result = self.generator(prompt, max_length=max_length)[0]
        return result["generated_text"]
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        max_length: int = 256,
    ) -> dict[str, Any]:
        """Full RAG pipeline: retrieve and generate."""
        # Retrieve
        retrieved = self.retrieve(question, top_k)
        context = [doc["text"] for doc in retrieved]
        
        # Generate
        answer = self.generate(question, context, max_length)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
        }


# Usage
rag = RAGSystem()
rag.add_documents(knowledge_base)
result = rag.query("What is machine learning?")
```

### Document Chunking for RAG

```python
from typing import Iterator

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = "\n\n",
) -> list[str]:
    """Split text into overlapping chunks."""
    # Split by separator first
    paragraphs = text.split(separator)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para.split())
        
        if current_size + para_size > chunk_size and current_chunk:
            chunks.append(separator.join(current_chunk))
            
            # Keep overlap
            overlap_size = 0
            overlap_start = len(current_chunk)
            for i in range(len(current_chunk) - 1, -1, -1):
                overlap_size += len(current_chunk[i].split())
                if overlap_size >= chunk_overlap:
                    overlap_start = i
                    break
            
            current_chunk = current_chunk[overlap_start:]
            current_size = sum(len(p.split()) for p in current_chunk)
        
        current_chunk.append(para)
        current_size += para_size
    
    if current_chunk:
        chunks.append(separator.join(current_chunk))
    
    return chunks
```
