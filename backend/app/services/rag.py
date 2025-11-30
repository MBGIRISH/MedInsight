try:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback for newer langchain versions
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        Chroma = None
        HuggingFaceEmbeddings = None

from typing import List, Dict, Any
import os


class RAGService:
    """RAG service with LLM-enhanced reasoning for medical knowledge retrieval."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        if HuggingFaceEmbeddings is None:
            raise ImportError("HuggingFaceEmbeddings not available. Install langchain-community.")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vectorstore = None
        self._initialize_vectorstore()
        
        # Initialize LLM service for reasoning
        try:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService()
        except Exception as e:
            print(f"Warning: Could not initialize LLM service: {e}")
            self.llm_service = None

    def _initialize_vectorstore(self):
        """Initialize or load the vector store."""
        if Chroma is None:
            print("Warning: Chroma not available. RAG will not work.")
            return
        try:
            # Try to load existing vectorstore
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                # Create new empty vectorstore
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            print(f"Warning: Could not load vectorstore: {e}")
            # Create new empty vectorstore
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except:
                print("Could not initialize vectorstore. RAG features will be limited.")
                self.vectorstore = None

    def _reformulate_query(self, query: str, query_type: str = "general") -> str:
        """Reformulate query using LLM if available, otherwise use rule-based."""
        # Try LLM-based reformulation first
        if self.llm_service and self.llm_service.llm:
            try:
                prompt = f"""As a medical information retrieval expert, reformulate this query to better match medical literature and guidelines:

Original Query: {query}
Query Type: {query_type}

Provide a reformulated query that:
1. Uses medical terminology
2. Includes relevant synonyms
3. Matches how information appears in medical guidelines

Reformulated Query:"""
                
                reformulated = self.llm_service.generate(prompt, max_tokens=100)
                # Clean up LLM response
                reformulated = reformulated.strip()
                if reformulated and len(reformulated) > 10:
                    return reformulated
            except Exception as e:
                print(f"LLM query reformulation failed: {e}, using rule-based")
        
        # Rule-based fallback
        if query_type == "symptom":
            return f"medical emergency symptoms {query} red flag warning critical urgent"
        elif query_type == "lab":
            return f"laboratory test {query} normal range reference values abnormal critical"
        elif query_type == "drug":
            return f"medication drug {query} dosage guidelines interaction contraindication"
        else:
            return query

    def _expand_symptom_query(self, query: str) -> str:
        """Expand symptoms into related medical terms."""
        # Medical term expansion dictionary
        symptom_expansions = {
            'chest pain': 'chest pain angina myocardial infarction acute coronary syndrome',
            'shortness of breath': 'shortness of breath dyspnea respiratory distress hypoxia',
            'fever': 'fever pyrexia hyperthermia infection sepsis',
            'headache': 'headache cephalgia migraine neurological emergency',
            'nausea': 'nausea vomiting emesis gastrointestinal',
            'dizziness': 'dizziness vertigo syncope neurological',
            'sweating': 'sweating diaphoresis perspiration',
            'confusion': 'confusion altered mental status neurological',
            'seizure': 'seizure convulsion epilepsy neurological emergency',
            'stroke': 'stroke CVA cerebrovascular accident neurological emergency',
            'diabetes': 'diabetes diabetic hyperglycemia hypoglycemia DKA HHS',
            'hypertension': 'hypertension high blood pressure hypertensive crisis',
            'meningitis': 'meningitis neck stiffness photophobia neurological infection',
            'sepsis': 'sepsis septic shock infection SIRS',
            'respiratory': 'respiratory failure hypoxia dyspnea breathing difficulty'
        }
        
        query_lower = query.lower()
        expanded_terms = [query]
        
        # Add related terms
        for symptom, expansion in symptom_expansions.items():
            if symptom in query_lower:
                expanded_terms.extend(expansion.split())
        
        return ' '.join(expanded_terms)

    def retrieve(self, query: str, top_k: int = 8, use_mmr: bool = True, query_type: str = "general") -> List[Dict[str, Any]]:
        """Retrieve relevant documents using MMR with k=8 and query expansion."""
        if not self.vectorstore:
            return []
        
        try:
            # Step 1: Expand query if symptom-based
            if query_type == "symptom":
                expanded_query = self._expand_symptom_query(query)
            else:
                expanded_query = query
            
            # Step 2: Reformulate query
            reformulated_query = self._reformulate_query(expanded_query, query_type)
            
            # Step 3: Always use MMR with k=8 for better diversity
            docs = self.vectorstore.max_marginal_relevance_search(
                reformulated_query,
                k=top_k,
                fetch_k=top_k * 3,  # Fetch more for better diversity
                lambda_mult=0.5  # Balance between relevance and diversity
            )
            
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
            
            # Step 4: Use custom prompt for retrieval reasoning
            if self.llm_service and self.llm_service.llm and results:
                try:
                    reasoning = self._custom_retrieval_reasoning(reformulated_query, results)
                    # Add reasoning to first result
                    if results:
                        results[0]['llm_reasoning'] = reasoning
                except Exception as e:
                    print(f"LLM reasoning failed: {e}")
            
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            # Fallback to similarity search
            try:
                docs = self.vectorstore.similarity_search(reformulated_query, k=top_k)
                results = []
                for doc in docs:
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                    })
                return results
            except:
                return []

    def _custom_retrieval_reasoning(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Use custom prompt for retrieval reasoning."""
        if not retrieved_docs:
            return "No relevant documents retrieved."
        
        context = "\n\n".join([f"Document {i+1}:\n{doc.get('content', '')[:400]}" 
                               for i, doc in enumerate(retrieved_docs[:3])])
        
        custom_prompt = f"""As a medical expert analyzing retrieved clinical guidelines, provide reasoning about the following:

Query: {query}

Retrieved Medical Guidelines:
{context}

Based on these guidelines, provide:
1. Clinical significance of the findings
2. Urgency level (critical/high/moderate/low)
3. Recommended immediate actions
4. Key evidence from guidelines

Medical Reasoning:"""
        
        if self.llm_service and self.llm_service.llm:
            return self.llm_service.generate(custom_prompt, max_tokens=300)
        else:
            return "Retrieved guidelines require medical review. Consult clinical protocols."

    def retrieve_dosage_guidelines(self, drug_name: str) -> List[Dict[str, Any]]:
        """Retrieve dosage guidelines for a specific drug."""
        query = f"dosage guidelines for {drug_name} maximum minimum recommended"
        return self.retrieve(query, top_k=8, query_type="drug")

    def retrieve_drug_interactions(self, drug1: str, drug2: str) -> List[Dict[str, Any]]:
        """Retrieve drug interaction information."""
        query = f"interaction between {drug1} and {drug2} contraindication"
        return self.retrieve(query, top_k=8, query_type="drug")

    def retrieve_red_flags(self, symptom: str) -> List[Dict[str, Any]]:
        """Retrieve red flag information for symptoms."""
        query = f"red flag emergency critical {symptom} warning"
        return self.retrieve(query, top_k=8, query_type="symptom")

    def retrieve_lab_ranges(self, test_name: str) -> List[Dict[str, Any]]:
        """Retrieve lab reference ranges."""
        query = f"normal range reference values for {test_name} lab test"
        return self.retrieve(query, top_k=8, query_type="lab")

    def retrieve_guidelines(self, condition: str) -> List[Dict[str, Any]]:
        """Retrieve treatment guidelines for a condition."""
        query = f"treatment guidelines for {condition} WHO NHS protocol"
        return self.retrieve(query, top_k=8, query_type="general")

