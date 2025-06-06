"""
Working MedGemma Biomarker Analyzer - Uses Greedy Decoding
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

class WorkingBiomarkerAnalyzer:
    """
    MedGemma Biomarker Analyzer that avoids CUDA sampling issues
    Uses greedy decoding for reliable generation
    """
    
    def __init__(self):
        self.model_name = "google/medgemma-4b-it"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
        # Create outputs directory
        Path("data/outputs").mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load MedGemma with working configuration"""
        if self.model_loaded:
            print("Model already loaded!")
            return
        
        print(f"Loading MedGemma on {self.device}...")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.model_loaded = True
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Test generation
        self._test_generation()
    
    def _test_generation(self):
        """Test that generation works"""
        test_prompt = "BRCA1 is"
        inputs = self.tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Greedy only
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test generation: '{response}'")
    
    def _generate_text(self, prompt, max_tokens=300):
        """Generate text using reliable greedy decoding"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1500
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding - reliable
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        return response
    
    def analyze_biomarkers(self, genes, context="", analysis_type="comprehensive"):
        """Analyze genes as biomarkers"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Handle file input
        if isinstance(genes, str) and genes.endswith('.csv'):
            genes = self.load_genes_from_file(genes)
        elif isinstance(genes, str):
            genes = [genes]
        
        # Clean gene names
        genes = [gene.strip().upper() for gene in genes if gene.strip()]
        
        # Create prompt
        prompt = self._create_prompt(genes, context, analysis_type)
        
        print(f"Analyzing {len(genes)} genes: {', '.join(genes[:5])}{'...' if len(genes) > 5 else ''}")
        if context:
            print(f"Context: {context}")
        
        # Generate analysis
        start_time = time.time()
        analysis = self._generate_text(prompt, max_tokens=400)
        gen_time = time.time() - start_time
        
        print(f"✅ Analysis completed in {gen_time:.2f} seconds")
        
        # Structure results
        result = {
            "timestamp": datetime.now().isoformat(),
            "genes": genes,
            "gene_count": len(genes),
            "context": context,
            "analysis_type": analysis_type,
            "analysis": analysis,
            "generation_time": gen_time
        }
        
        return result
    
    def _create_prompt(self, genes, context, analysis_type):
        """Create biomarker analysis prompt"""
        gene_list = ", ".join(genes[:20])  # Limit for prompt length
        if len(genes) > 20:
            gene_list += f" and {len(genes) - 20} additional genes"
        
        context_text = f" in the context of {context}" if context else ""
        
        prompt = f"""As a medical AI assistant, analyze these genes as biomarkers{context_text}: {gene_list}

Please provide:

1. BIOMARKER CLASSIFICATION: Type of biomarker (diagnostic/prognostic/predictive/monitoring)
2. CLINICAL UTILITY: Current medical applications and evidence level
3. DISEASE ASSOCIATIONS: Primary diseases and conditions linked to these genes
4. THERAPEUTIC IMPLICATIONS: Treatment guidance and personalized medicine applications
5. CLINICAL INTERPRETATION: How to interpret results in patient care

Focus on evidence-based medical information and clinical actionability.

Biomarker Analysis:
"""
        return prompt
    
    def load_genes_from_file(self, file_path):
        """Load genes from CSV file"""
        df = pd.read_csv(file_path)
        genes = df.iloc[:, 0].dropna().astype(str).tolist()
        print(f"Loaded {len(genes)} genes from {file_path}")
        return genes
    
    def save_analysis(self, result, filename=None):
        """Save analysis to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biomarker_analysis_{timestamp}.json"
        
        output_path = Path("data/outputs") / filename
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Analysis saved to: {output_path}")
        return str(output_path)
    
    def analyze_gene_file(self, file_path, context="", save_results=True):
        """Analyze genes from file and optionally save results"""
        result = self.analyze_biomarkers(file_path, context)
        
        if save_results:
            save_path = self.save_analysis(result)
            result["saved_to"] = save_path
        
        return result

# Test the working analyzer
def test_working_analyzer():
    print("🧬 Testing Working Biomarker Analyzer")
    print("=" * 50)
    
    analyzer = WorkingBiomarkerAnalyzer()
    analyzer.load_model()
    
    # Test with a few genes
    test_genes = ["BRCA1", "BRCA2", "TP53"]
    result = analyzer.analyze_biomarkers(
        genes=test_genes,
        context="breast cancer risk assessment"
    )
    
    print("\n📋 ANALYSIS RESULTS:")
    print("=" * 40)
    print(result['analysis'])
    print("=" * 40)
    
    # Test with your gene file
    gene_file = "data/gene_lists/gene_list2.csv"
    if Path(gene_file).exists():
        print(f"\n📊 Analyzing gene file: {gene_file}")
        file_result = analyzer.analyze_gene_file(
            gene_file, 
            context="cardiovascular disease biomarkers"
        )
        
        print(f"✅ Analyzed {file_result['gene_count']} genes")
        print(f"💾 Saved to: {file_result.get('saved_to', 'Not saved')}")

if __name__ == "__main__":
    test_working_analyzer()