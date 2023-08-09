mkdir -p data && cd data
wget 'https://huggingface.co/michaelfeil/ct2fast-WizardCoder-15B-V1.0/resolve/8dc8b3268d136bae06b91102c99ac6f4ac4579a4/model.bin'
wget 'https://huggingface.co/michaelfeil/ct2fast-WizardCoder-15B-V1.0/resolve/8dc8b3268d136bae06b91102c99ac6f4ac4579a4/config.json'
wget 'https://huggingface.co/michaelfeil/ct2fast-WizardCoder-15B-V1.0/resolve/8dc8b3268d136bae06b91102c99ac6f4ac4579a4/vocabulary.json'

mkdir -p tokenizer && cd tokenizer
wget 'https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/4d56635bd60520e49c74c48a96d644854af1c712/config.json'
wget 'https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/resolve/4d56635bd60520e49c74c48a96d644854af1c712/tokenizer.json'
