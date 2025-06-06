"""
MedGemma Biomarker Analyzer - Main Analysis Module
Author: Arian Abdi
Project: medgemma-biomarkers
Purpose: Comprehensive gene biomarker analysis using Google's MedGemma model

This module provides:
- Gene biomarker classification and analysis
- Clinical interpretation and recommendations
- Multiple analysis types (diagnostic, prognostic, predictive)
- Batch processing capabilities
- Structured output formats
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class BiomarkerAnalyzer:
    """
    Main class for comprehensive biomarker analysis using MedGemma
    
    Features:
    - Multiple analysis types (comprehensive, diagnostic, prognostic, predictive)
    - Flexible input methods (lists, files)
    - GPU optimization with quantization
    - Structured output with timestamps
    - Clinical context integration
    """
    
    def __init__(self, model_name: str = "google/medgemma-4b-it", use_gpu: bool = True):
        """
        Initialize the BiomarkerAnalyzer
        
        Args:
            model_name: HuggingFace model identifier
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        self.model_loaded = False
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        self.output_dir = Path("data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"BiomarkerAnalyzer initialized for device: {self.device}")
    
    def _setup_logging(self):
        """Configure logging for the analyzer"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('biomarker_analysis.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, use_quantization: bool = True, force_reload: bool = False):
        """
        Load MedGemma model with optimization
        
        Args:
            use_quantization: Use 4-bit quantization for memory efficiency
            force_reload: Force reload even if already loaded
        """
        if self.model_loaded and not force_reload:
            self.logger.info("Model already loaded. Use force_reload=True to reload.")
            return self.tokenizer, self.model
        
        self.logger.info(f"Loading {self.model_name} on {self.device}")
        start_time = time.time()
        
        try:
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Model configuration
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # Add quantization for memory efficiency
            if use_quantization and self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.logger.info("Using 4-bit quantization for memory efficiency")
                except ImportError:
                    self.logger.warning("BitsAndBytesConfig not available, loading without quantization")
            
            # Load model
            self.logger.info("Loading MedGemma model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            load_time = time.time() - start_time
            self.model_loaded = True
            self.logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            
            return self.tokenizer, self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_genes_from_file(self, file_path: str) -> List[str]:
        """
        Load gene list from CSV or TXT file
        
        Args:
            file_path: Path to gene list file
            
        Returns:
            List of gene names
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Gene list file not found: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            # Read CSV file - assume first column contains gene names
            df = pd.read_csv(file_path)
            genes = df.iloc[:, 0].dropna().astype(str).tolist()
            self.logger.info(f"Loaded {len(genes)} genes from CSV: {file_path.name}")
            
        elif file_path.suffix.lower() == '.txt':
            # Read text file - one gene per line
            with open(file_path, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(genes)} genes from TXT: {file_path.name}")
            
        else:
            raise ValueError("File must be .csv or .txt format")
        
        # Clean gene names
        genes = [gene.strip().upper() for gene in genes if gene.strip()]
        
        return genes
    
    def analyze_biomarkers(self, 
                          genes: Union[List[str], str], 
                          context: str = "", 
                          analysis_type: str = "comprehensive",
                          max_genes_per_batch: int = 20) -> Dict:
        """
        Analyze genes as biomarkers
        
        Args:
            genes: List of gene names or path to gene file
            context: Clinical context (e.g., "breast cancer", "cardiovascular disease")
            analysis_type: Type of analysis ("comprehensive", "diagnostic", "prognostic", "predictive")
            max_genes_per_batch: Maximum genes to analyze in one prompt
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.model_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded")
        
        # Handle file input
        if isinstance(genes, str):
            genes = self.load_genes_from_file(genes)
        
        # Clean and validate genes
        genes = [gene.strip().upper() for gene in genes if gene.strip()]
        
        if not genes:
            raise ValueError("No valid genes provided")
        
        # For large gene lists, analyze in batches
        if len(genes) > max_genes_per_batch:
            return self._analyze_large_gene_set(genes, context, analysis_type, max_genes_per_batch)
        
        # Create analysis prompt
        prompt = self._create_biomarker_prompt(genes, context, analysis_type)
        
        # Generate analysis
        self.logger.info(f"Analyzing {len(genes)} genes with context: '{context}'")
        response = self._generate_response(prompt)
        
        # Structure results
        result = {
            "analysis_id": f"analysis_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "genes": genes,
            "gene_count": len(genes),
            "context": context,
            "analysis_type": analysis_type,
            "analysis": response,
            "model_info": {
                "model_name": self.model_name,
                "device": self.device
            }
        }
        
        return result
    
    def _analyze_large_gene_set(self, genes: List[str], context: str, 
                               analysis_type: str, batch_size: int) -> Dict:
        """
        Analyze large gene sets by batching
        
        Args:
            genes: List of gene names
            context: Clinical context
            analysis_type: Type of analysis
            batch_size: Genes per batch
            
        Returns:
            Combined analysis results
        """
        self.logger.info(f"Large gene set detected ({len(genes)} genes). Processing in batches of {batch_size}")
        
        # Split genes into batches
        gene_batches = [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]
        batch_results = []
        
        for i, batch in enumerate(gene_batches, 1):
            self.logger.info(f"Processing batch {i}/{len(gene_batches)} ({len(batch)} genes)")
            
            batch_result = self.analyze_biomarkers(
                genes=batch,
                context=context,
                analysis_type=analysis_type,
                max_genes_per_batch=batch_size
            )
            batch_results.append(batch_result)
            
            # Small delay between batches
            time.sleep(1)
        
        # Combine results
        combined_result = {
            "analysis_id": f"batch_analysis_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "genes": genes,
            "gene_count": len(genes),
            "context": context,
            "analysis_type": analysis_type,
            "batch_count": len(gene_batches),
            "batch_results": batch_results,
            "summary": self._generate_summary_analysis(genes, context, batch_results),
            "model_info": {
                "model_name": self.model_name,
                "device": self.device
            }
        }
        
        return combined_result
    
    def _generate_summary_analysis(self, genes: List[str], context: str, batch_results: List[Dict]) -> str:
        """Generate summary analysis for large gene sets"""
        
        gene_list = ", ".join(genes[:10])  # Show first 10 genes
        if len(genes) > 10:
            gene_list += f" ... and {len(genes) - 10} more"
        
        summary_prompt = f"""
Based on the analysis of {len(genes)} genes ({gene_list}) in the context of {context}, 
provide a comprehensive summary that includes:

1. OVERALL BIOMARKER PROFILE: Key patterns and themes across all genes
2. CLINICAL SIGNIFICANCE: Most important findings for clinical practice
3. PATHWAY CONNECTIONS: Major biological pathways represented
4. THERAPEUTIC IMPLICATIONS: Treatment and diagnostic opportunities
5. RESEARCH PRIORITIES: Areas needing further investigation

Focus on the most clinically actionable insights from this gene set.

Summary Analysis:
"""
        
        return self._generate_response(summary_prompt, max_tokens=400)
    
    def _create_biomarker_prompt(self, genes: List[str], context: str, analysis_type: str) -> str:
        """Create specialized prompts for different analysis types"""
        
        gene_list = ", ".join(genes)
        context_phrase = f"in the context of {context}" if context else "for general clinical applications"
        
        prompts = {
            "comprehensive": f"""
As a medical AI assistant specializing in biomarker analysis, provide a comprehensive evaluation of these genes as biomarkers: {gene_list}
{context_phrase.capitalize()}

Please analyze:

1. BIOMARKER CLASSIFICATION
   - Classify each gene as diagnostic, prognostic, predictive, or monitoring biomarker
   - Indicate confidence level and evidence strength

2. CLINICAL UTILITY
   - Current clinical applications and FDA approvals
   - Evidence level (Level I-IV) for each application
   - Recommended testing scenarios

3. GENE RELATIONSHIPS
   - Functional connections and pathway interactions
   - Synergistic or antagonistic relationships
   - Co-expression patterns

4. DISEASE ASSOCIATIONS
   - Primary diseases and conditions
   - Risk stratification capabilities
   - Population-specific considerations

5. CLINICAL INTERPRETATION
   - How to interpret positive/negative results
   - Cut-off values and reference ranges where applicable
   - Clinical decision-making guidance

6. ACTIONABLE INSIGHTS
   - Therapeutic implications and treatment selection
   - Monitoring strategies
   - Personalized medicine applications

7. LIMITATIONS AND CONSIDERATIONS
   - Technical limitations
   - Population bias and validation needs
   - Cost-effectiveness considerations

Biomarker Analysis:
""",
            
            "diagnostic": f"""
Evaluate these genes as diagnostic biomarkers: {gene_list}
{context_phrase.capitalize()}

Focus on:
1. DIAGNOSTIC PERFORMANCE: Sensitivity, specificity, PPV, NPV
2. CLINICAL APPLICATION: When and how to use for diagnosis
3. INTERPRETATION: Result interpretation and clinical significance
4. COMPLEMENTARY TESTS: Other biomarkers or tests to consider
5. LIMITATIONS: False positives/negatives and interfering factors

Diagnostic Biomarker Analysis:
""",
            
            "prognostic": f"""
Assess these genes as prognostic biomarkers: {gene_list}
{context_phrase.capitalize()}

Analyze:
1. PROGNOSTIC VALUE: Survival outcomes and disease progression
2. RISK STRATIFICATION: Patient categorization and risk levels
3. TEMPORAL ASPECTS: Short-term vs long-term prognostic value
4. CLINICAL DECISIONS: Treatment planning based on prognosis
5. VALIDATION: Study evidence and population validation

Prognostic Analysis:
""",
            
            "predictive": f"""
Evaluate these genes as predictive biomarkers: {gene_list}
{context_phrase.capitalize()}

Focus on:
1. TREATMENT RESPONSE: Prediction of therapeutic efficacy
2. DRUG SELECTION: Personalized therapy recommendations
3. RESISTANCE MECHANISMS: Potential treatment resistance
4. MONITORING: Treatment response monitoring strategies
5. COMPANION DIAGNOSTICS: FDA-approved test relationships

Predictive Biomarker Analysis:
"""
        }
        
        return prompts.get(analysis_type, prompts["comprehensive"])
    
    def _generate_response(self, prompt: str, max_tokens: int = 600) -> str:
        """Generate response from MedGemma"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9
            )
        
        generation_time = time.time() - start_time
        
        # Extract generated text
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        self.logger.info(f"Response generated in {generation_time:.2f} seconds")
        return response
    
    def save_analysis(self, result: Dict, filename: str = None) -> str:
        """
        Save analysis results to file
        
        Args:
            result: Analysis result dictionary
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biomarker_analysis_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Analysis saved to: {output_path}")
        return str(output_path)
    
    def analyze_gene_list_file(self, file_path: str, context: str = "", 
                              analysis_type: str = "comprehensive", 
                              save_results: bool = True) -> Dict:
        """
        Convenience method to analyze genes from file and save results
        
        Args:
            file_path: Path to gene list file
            context: Clinical context
            analysis_type: Type of analysis
            save_results: Whether to save results automatically
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Starting analysis of gene file: {file_path}")
        
        # Perform analysis
        result = self.analyze_biomarkers(
            genes=file_path,
            context=context,
            analysis_type=analysis_type
        )
        
        # Save results if requested
        if save_results:
            save_path = self.save_analysis(result)
            result["saved_to"] = save_path
        
        return result

# Convenience functions for quick analysis
def analyze_gene_file(file_path: str, context: str = "", analysis_type: str = "comprehensive"):
    """Quick analysis of gene file"""
    analyzer = BiomarkerAnalyzer()
    analyzer.load_model()
    return analyzer.analyze_gene_list_file(file_path, context, analysis_type)

def quick_gene_analysis(genes: List[str], context: str = ""):
    """Quick analysis of gene list"""
    analyzer = BiomarkerAnalyzer()
    analyzer.load_model()
    return analyzer.analyze_biomarkers(genes, context)

if __name__ == "__main__":
    # Example usage with your gene list files
    print("🧬 MedGemma Biomarker Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BiomarkerAnalyzer()
    
    # Load model
    print("Loading MedGemma model...")
    analyzer.load_model()
    
    # Analyze your gene lists
    gene_files = [
        "data/gene_lists/gene_list1.csv",  # 303 genes
        "data/gene_lists/gene_list2.csv",  # 83 genes  
        "data/gene_lists/gene_list3.csv"   # 115 genes
    ]
    
    for file_path in gene_files:
        if Path(file_path).exists():
            print(f"\n📊 Analyzing: {file_path}")
            result = analyzer.analyze_gene_list_file(
                file_path=file_path,
                context="general biomarker screening",
                analysis_type="comprehensive"
            )
            print(f"✅ Analysis completed for {result['gene_count']} genes")
            print(f"💾 Results saved to: {result.get('saved_to', 'Not saved')}")
        else:
            print(f"❌ File not found: {file_path}")
    
    print("\n🎉 All analyses completed!")