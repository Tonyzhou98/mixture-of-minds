from transformers import Gemma3ForCausalLM, Gemma3TextConfig, AutoTokenizer
# according to the issue https://github.com/volcengine/verl/issues/1013#issuecomment-2812388104
# making a verl trainable version for gemma3
#  srun --gres=gpu:8 --mem 128G -c 64 python convert_gemma3.py

# load gemma3 model
local_path = "/your/path/to/your_fs/models/gemma-3-27b-it" 
config = Gemma3TextConfig.from_pretrained(local_path)
tokenizer = AutoTokenizer.from_pretrained(local_path)
config.architectures = ["Gemma3ForCausalLM"]
model = Gemma3ForCausalLM.from_pretrained(local_path, config= config, trust_remote_code=True)
print("model loaded")
# Now save
save_path = "/your/path/to/your_fs/models/gemma3_27b_textonly"
print("saving model to", save_path)
model.save_pretrained(save_path)
config.save_pretrained(save_path) 
tokenizer.save_pretrained(save_path)
print("model saved")
