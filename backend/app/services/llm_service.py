"""
LLM Service with support for medical-capable models and fallback logic.
Supports: GPT-4o, GPT-4.1, BioGPT, ClinicalBERT, MedAlpaca, Meditron
"""
from typing import Optional, List, Dict, Any
import os
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars


class LLMService:
    """Medical LLM service with multiple model support and fallback."""
    
    def __init__(self):
        self.llm = None
        self.model_type = None
        self.model_name = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM with fallback chain - uses any available good models."""
        # Priority order: GPT-4o > GPT-4 Turbo > GPT-4 > GPT-3.5 Turbo > Small HF Models > Meditron > MedAlpaca > BioGPT > ClinicalBERT
        
        # Try OpenAI models first (if API key available)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            # Try OpenAI GPT-4o first
            if self._try_openai_gpt4o():
                return
            
            # Try OpenAI GPT-4 Turbo
            if self._try_openai_gpt4_turbo():
                return
            
            # Try OpenAI GPT-4
            if self._try_openai_gpt4():
                return
            
            # Try OpenAI GPT-3.5 Turbo (more reliable, less quota issues)
            if self._try_openai_gpt3_5_turbo():
                return
        
        # Try HuggingFace models (no API key needed, but may require GPU)
        print("Trying HuggingFace models (no API key required)...")
        
        # Try smaller, faster models first (work on CPU, load quickly)
        if self._try_huggingface_small_model():
            return
        
        # Try HuggingFace Meditron (if available)
        if self._try_huggingface_meditron():
            return
        
        # Try HuggingFace MedAlpaca (if available)
        if self._try_huggingface_medalpaca():
            return
        
        # Try HuggingFace BioGPT (if available)
        if self._try_huggingface_biogpt():
            return
        
        # Try ClinicalBERT as last resort
        if self._try_clinicalbert():
            return
        
        # No LLM available - will use rule-based fallback
        print("Warning: No LLM available. Using rule-based reasoning only.")
        self.llm = None
    
    def _try_openai_gpt4o(self) -> bool:
        """Try to initialize OpenAI GPT-4o."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("No OPENAI_API_KEY found in environment")
                return False
            
            print(f"Found OpenAI API key, attempting to initialize GPT-4o...")
            
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=api_key
                )
                # Don't test during init to avoid quota issues - test will happen on first use
                self.model_type = "openai"
                self.model_name = "gpt-4o"
                print("✅ Using OpenAI GPT-4o for medical reasoning")
                return True
            except ImportError:
                print("langchain_openai not installed, trying alternative...")
                # Try alternative import
                try:
                    from langchain.chat_models import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model_name="gpt-4o",
                        temperature=0.1,
                        max_tokens=1000,
                        openai_api_key=api_key
                    )
                    self.model_type = "openai"
                    self.model_name = "gpt-4o"
                    print("✅ Using OpenAI GPT-4o for medical reasoning")
                    return True
                except Exception as e:
                    print(f"Could not initialize GPT-4o with alternative import: {e}")
                    return False
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg:
                    print(f"OpenAI quota exceeded for GPT-4o, trying other models...")
                else:
                    print(f"Could not initialize GPT-4o: {e}")
                return False
        except Exception as e:
            print(f"Error in _try_openai_gpt4o: {e}")
            return False
    
    def _try_openai_gpt4_turbo(self) -> bool:
        """Try to initialize OpenAI GPT-4 Turbo."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False
            
            try:
                from langchain_openai import ChatOpenAI
                # Try current GPT-4 Turbo model names (corrected)
                # Note: "gpt-4-turbo" doesn't exist, use valid model names
                # Skip testing to avoid quota issues during initialization
                for model in ["gpt-4-turbo-preview", "gpt-4-0125-preview", "gpt-4-1106-preview"]:
                    try:
                        self.llm = ChatOpenAI(
                            model=model,
                            temperature=0.1,
                            max_tokens=1000,
                            api_key=api_key
                        )
                        # Don't test during init to avoid quota issues - test will happen on first use
                        self.model_type = "openai"
                        self.model_name = model
                        print(f"✅ Using OpenAI {model} for medical reasoning")
                        return True
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "quota" in error_msg or "429" in error_msg:
                            print(f"OpenAI quota exceeded for {model}, trying next model...")
                        elif "404" in error_msg or "not found" in error_msg or "does not exist" in error_msg:
                            print(f"Model {model} not found, trying next model...")
                        else:
                            print(f"Could not initialize {model}: {e}")
                        continue
                return False
            except ImportError:
                return False
        except Exception as e:
            return False
    
    def _try_openai_gpt4(self) -> bool:
        """Try to initialize OpenAI GPT-4."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False
            
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=api_key
                )
                self.model_type = "openai"
                self.model_name = "gpt-4"
                print("✅ Using OpenAI GPT-4 for medical reasoning")
                return True
            except ImportError:
                return False
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg:
                    print(f"OpenAI quota exceeded for GPT-4, trying next model...")
                return False
        except Exception as e:
            return False
    
    def _try_openai_gpt3_5_turbo(self) -> bool:
        """Try to initialize OpenAI GPT-3.5 Turbo."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return False
            
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=1000,
                    api_key=api_key
                )
                self.model_type = "openai"
                self.model_name = "gpt-3.5-turbo"
                print("✅ Using OpenAI GPT-3.5 Turbo for medical reasoning")
                return True
            except ImportError:
                return False
            except Exception as e:
                error_msg = str(e).lower()
                if "quota" in error_msg or "429" in error_msg:
                    print(f"OpenAI quota exceeded for GPT-3.5 Turbo, trying HuggingFace models...")
                return False
        except Exception as e:
            return False
    
    def _try_huggingface_small_model(self) -> bool:
        """Try to initialize a small, fast HuggingFace model that works well for medical reasoning."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            # Try smaller models first (faster, less memory, work on CPU)
            small_models = [
                "gpt2",  # Very fast, widely available, works on CPU
                "distilgpt2",  # Even faster
            ]
            
            for model_name in small_models:
                try:
                    print(f"Trying {model_name}...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    # Add padding token if not present
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=200,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    from langchain_huggingface import HuggingFacePipeline
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                    self.model_type = "huggingface"
                    self.model_name = model_name
                    print(f"✅ Using HuggingFace {model_name} for medical reasoning")
                    return True
                except Exception as e:
                    print(f"Could not load {model_name}: {e}")
                    continue
            
            return False
        except ImportError as ie:
            print(f"Could not import transformers: {ie}")
            return False
        except Exception as e:
            print(f"Could not initialize small HuggingFace model: {e}")
            return False
    
    def _try_huggingface_meditron(self) -> bool:
        """Try to initialize HuggingFace Meditron."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            import os
            
            # Check if user has HuggingFace token for gated models
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            
            # Try smaller open models first
            model_names = [
                "microsoft/biogpt-large",  # Alternative medical model
                "epfl-llm/meditron-7b"  # Requires authentication
            ]
            
            for model_name in model_names:
                try:
                    # Skip gated models if no token
                    if "meditron" in model_name and not hf_token:
                        continue
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=hf_token if hf_token else None
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        token=hf_token if hf_token else None,
                        low_cpu_mem_usage=True
                    )
                    
                    from langchain.llms import HuggingFacePipeline
                    from transformers import pipeline
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=500,
                        temperature=0.1,
                        do_sample=True
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                    self.model_type = "huggingface"
                    self.model_name = model_name.split("/")[-1]
                    print(f"✅ Using HuggingFace {self.model_name} for medical reasoning")
                    return True
                except Exception as e:
                    if "gated" not in str(e).lower() and "access" not in str(e).lower():
                        print(f"Could not load {model_name}: {e}")
                    continue
            return False
        except Exception as e:
            return False
    
    def _try_huggingface_medalpaca(self) -> bool:
        """Try to initialize HuggingFace MedAlpaca."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Try smaller models first
            model_names = [
                "medalpaca/medalpaca-7b",
                "medalpaca/medalpaca-13b"
            ]
            
            for model_name in model_names:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # Use slow tokenizer
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        low_cpu_mem_usage=True
                    )
                    
                    from langchain.llms import HuggingFacePipeline
                    from transformers import pipeline
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=500,
                        temperature=0.1,
                        do_sample=True
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=pipe)
                    self.model_type = "huggingface"
                    self.model_name = model_name.split("/")[-1]
                    print(f"✅ Using HuggingFace {self.model_name} for medical reasoning")
                    return True
                except Exception as e:
                    if "tokenizer" not in str(e).lower():
                        print(f"Could not load {model_name}: {e}")
                    continue
            return False
        except Exception as e:
            return False
    
    def _try_huggingface_biogpt(self) -> bool:
        """Try to initialize HuggingFace BioGPT."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "microsoft/biogpt"
            
            try:
                # Install sacremoses if needed
                try:
                    import sacremoses
                except ImportError:
                    print("Installing sacremoses for BioGPT...")
                    import subprocess
                    subprocess.check_call(["pip", "install", "sacremoses"])
                    import sacremoses
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                try:
                    from langchain_community.llms import HuggingFacePipeline
                except ImportError:
                    try:
                        from langchain.llms import HuggingFacePipeline
                    except ImportError:
                        print("Could not import HuggingFacePipeline")
                        return False
                from transformers import pipeline
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=True
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                self.model_type = "huggingface"
                self.model_name = "biogpt"
                print("✅ Using HuggingFace BioGPT for medical reasoning")
                return True
            except Exception as e:
                print(f"Could not load BioGPT: {e}")
                return False
        except Exception as e:
            return False
    
    def _try_clinicalbert(self) -> bool:
        """Try to initialize ClinicalBERT (for embeddings/reasoning)."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                
                # ClinicalBERT is for embeddings, but we can use it for simple reasoning
                # For actual generation, we'd need a different approach
                self.model_type = "huggingface"
                self.model_name = "clinicalbert"
                print("✅ Using ClinicalBERT (limited - embeddings only)")
                # Note: ClinicalBERT is encoder-only, so limited for generation
                return False  # Don't use as primary LLM
            except Exception as e:
                return False
        except Exception as e:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using the initialized LLM."""
        if not self.llm:
            return self._rule_based_fallback(prompt)
        
        try:
            if self.model_type == "openai":
                # OpenAI ChatOpenAI
                try:
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    return str(response)
                except Exception as e:
                    error_msg = str(e).lower()
                    # If model doesn't exist or quota exceeded, try to reinitialize with fallback
                    if "404" in error_msg or "not found" in error_msg or "does not exist" in error_msg:
                        print(f"Model {self.model_name} not available, trying fallback models...")
                        # Try GPT-3.5 Turbo as fallback
                        if self._try_openai_gpt3_5_turbo():
                            try:
                                response = self.llm.invoke(prompt)
                                if hasattr(response, 'content'):
                                    return response.content
                                return str(response)
                            except:
                                pass  # Fall through to rule-based
                    elif "quota" in error_msg or "429" in error_msg:
                        print(f"OpenAI quota exceeded, using rule-based fallback")
                    # Always fall back to rule-based, don't raise
                    return self._rule_based_fallback(prompt)
            else:
                # HuggingFace pipeline - use invoke or __call__
                try:
                    response = self.llm.invoke(prompt)
                except:
                    try:
                        response = self.llm(prompt)
                    except:
                        # Try direct pipeline call
                        if hasattr(self.llm, 'pipeline'):
                            response = self.llm.pipeline(prompt, max_length=max_tokens, return_full_text=False)
                        else:
                            raise Exception("Could not call HuggingFace pipeline")
                
                if isinstance(response, list) and len(response) > 0:
                    if isinstance(response[0], dict):
                        return response[0].get('generated_text', str(response[0]))
                    return str(response[0])
                elif isinstance(response, str):
                    return response
                return str(response)
        except Exception as e:
            error_msg = str(e).lower()
            # Don't print full error for quota/404 issues (already handled)
            if "404" not in error_msg and "not found" not in error_msg and "quota" not in error_msg:
                print(f"LLM generation error: {e}, using fallback")
            return self._rule_based_fallback(prompt)
    
    def _rule_based_fallback(self, prompt: str) -> str:
        """Rule-based fallback when LLM is unavailable."""
        prompt_lower = prompt.lower()
        
        # Medical reasoning patterns
        if "dosage" in prompt_lower and "safe" in prompt_lower:
            return "Dosage safety requires checking against maximum recommended doses and patient-specific factors like age, weight, and comorbidities."
        elif "interaction" in prompt_lower:
            return "Drug interactions can cause increased toxicity, reduced efficacy, or life-threatening adverse reactions. Check interaction databases."
        elif "red flag" in prompt_lower or "emergency" in prompt_lower:
            return "Red flag symptoms indicate potential medical emergencies requiring immediate evaluation. Common patterns include chest pain with SOB, high glucose with symptoms, or neurological deficits."
        elif "lab" in prompt_lower or "test" in prompt_lower:
            return "Lab values should be compared against normal reference ranges. Abnormal values may indicate underlying conditions requiring attention."
        else:
            return "Medical evaluation required. Consider consulting clinical guidelines and evidence-based protocols."
    
    def reason_about_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Use LLM to reason about retrieved documents."""
        if not retrieved_docs:
            return "No relevant documents retrieved."
        
        context = "\n".join([doc.get('content', '')[:300] for doc in retrieved_docs[:3]])
        
        prompt = f"""As a medical expert, analyze the following clinical information and retrieved medical guidelines:

Query: {query}

Retrieved Guidelines:
{context}

Based on this information, provide a concise medical reasoning about the clinical significance and recommended actions. Focus on:
1. What the retrieved guidelines indicate
2. Clinical significance
3. Recommended next steps

Reasoning:"""
        
        return self.generate(prompt, max_tokens=300)
    
    def evaluate_agent_decision(self, agent_name: str, context: str, decision: str) -> str:
        """Use LLM to evaluate and enhance agent decisions."""
        prompt = f"""As a medical expert, review this agent decision:

Agent: {agent_name}
Context: {context}
Current Decision: {decision}

Provide enhanced medical reasoning that:
1. Explains why this finding is clinically significant
2. Identifies potential risks or concerns
3. Suggests appropriate clinical actions

Enhanced Reasoning:"""
        
        return self.generate(prompt, max_tokens=200)

