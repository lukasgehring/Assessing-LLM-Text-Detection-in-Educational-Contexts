import torch
from tqdm import tqdm

from detectors.detect_gpt_based_detectors.detect_gpt import preprocess_data
from detectors.detector_interface import Detector
from detectors.utils.load_hf import hf_load_pretrained_llm


class FastDetectGPT(Detector):
    def __init__(self, args):
        super().__init__("fast-detect-gpt", args)

    def get_samples(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = 10000
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def get_likelihood(self, logits, labels):
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    def get_sampling_discrepancy(self, logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self.get_samples(logits_ref, labels)
        log_likelihood_x = self.get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self.get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
        return discrepancy.item()

    def run(self, data):
        # loading huggingface pretrained models. Add correct model_class to load the correct model!
        mask_model, mask_tokenizer = hf_load_pretrained_llm(self.args.mask_filling_model_name,
                                                            cache_dir=self.args.cache_dir, device_map=self.args.device)
        mask_model.eval()

        base_model, base_tokenizer = hf_load_pretrained_llm(self.args.base_model_name,
                                                            cache_dir=self.args.cache_dir, device_map=self.args.device)
        base_model.eval()

        # strip whitespaces, newlines and keep only samples with <= 512 token
        # data = preprocess_data(data, mask_tokenizer=mask_tokenizer, args=self.args)
        data = data.head(self.args.n_samples)

        self.add_data_hash_to_args(human_data=data[data.is_human == 1], llm_data=data[data.is_human == 0])

        results = []
        for text in tqdm(data.answer):
            # original text
            tokenized = base_tokenizer(text, return_tensors="pt", padding=True,
                                       return_token_type_ids=False).to(self.args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = base_model(**tokenized).logits[:, :-1]
                if self.args.mask_filling_model_name == self.args.base_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = mask_tokenizer(text, return_tensors="pt", padding=True,
                                               return_token_type_ids=False).to(self.args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = mask_model(**tokenized).logits[:, :-1]
                original_crit = self.get_sampling_discrepancy(logits_ref, logits_score, labels)
            # result
            results.append(original_crit)

        self.save(predictions=results, answer_ids=data.id)

        return results, data.id
