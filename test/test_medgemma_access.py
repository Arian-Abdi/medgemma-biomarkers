"""
Fixed MedGemma test with proper generation handling
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def test_medgemma_generation_fixed():
    print("🔍 Testing MedGemma Generation (Fixed)...")
    print("-" * 50)
    
    try:
        torch.cuda.empty_cache()
        
        # Load tokenizer
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
        
        # Set special tokens properly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"   ✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        
        # Load model
        print("2. Loading model with quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-4b-it",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("   ✅ Model loaded!")
        
        # Test generation with safer parameters
        print("3. Testing generation (conservative approach)...")
        
        test_prompt = "BRCA1 gene function:"
        
        inputs = tokenizer(
            test_prompt, 
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=20
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Input tokens: {inputs['input_ids'][0].tolist()}")
        
        # Generate with very conservative settings
        with torch.no_grad():
            # Use greedy decoding first (no sampling)
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=20,
                min_new_tokens=1,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                output_scores=False,
                return_dict_in_generate=False
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(test_prompt):].strip()
        
        print(f"   ✅ Generation successful!")
        print(f"   Full response: '{response}'")
        print(f"   Generated: '{generated}'")
        
        # Now try with sampling
        print("4. Testing with sampling...")
        
        with torch.no_grad():
            outputs_sample = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=15,
                min_new_tokens=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response_sample = tokenizer.decode(outputs_sample[0], skip_special_tokens=True)
        generated_sample = response_sample[len(test_prompt):].strip()
        
        print(f"   ✅ Sampling generation successful!")
        print(f"   Sampled response: '{generated_sample}'")
        
        print("\n🎉 SUCCESS: MedGemma generation working properly!")
        
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        
        # Debug information
        if 'inputs' in locals():
            print(f"Debug - Input IDs: {inputs['input_ids']}")
            print(f"Debug - Input shape: {inputs['input_ids'].shape}")
            print(f"Debug - Min ID: {inputs['input_ids'].min()}")
            print(f"Debug - Max ID: {inputs['input_ids'].max()}")
            print(f"Debug - Vocab size: {tokenizer.vocab_size}")
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_medgemma_generation_fixed()