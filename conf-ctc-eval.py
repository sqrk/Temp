import nemo.collections.asr as nemo_asr

config_path = "Configs/conformer_ctc_bpe_small.yaml"

from ruamel.yaml import YAML

yaml = YAML(typ="safe")
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params["model"]["test_ds"]["batch_size"] = 8
params["model"]["test_ds"]["manifest_filepath"] = main_dir + "Manifests/Nemo/VCTK-test-manifest.json"
params["model"]["test_ds"]["sample_rate"] = 48000

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small") #MODIF
model.cfg.sample_rate = 48000

model.setup_test_data(test_data_config=params['model']['test_ds'])
model.cuda()

print("CFG:", model.cfg)

wer_nums = []
wer_denoms = []

model._wer.use_cer = False #Toggle for CER

for test_batch in model.test_dataloader():
    test_batch = [x.cuda() for x in test_batch]
    targets = test_batch[2]
    targets_lengths = test_batch[3]        
    log_probs, encoded_len, greedy_predictions = model(
        input_signal=test_batch[0], input_signal_length=test_batch[1]
    )
    
    model._wer.update(greedy_predictions, targets, targets_lengths)
    _, wer_num, wer_denom = model._wer.compute()
    model._wer.reset()
    wer_nums.append(wer_num.detach().cpu().numpy())
    wer_denoms.append(wer_denom.detach().cpu().numpy())

    del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

print(f"WER = {sum(wer_nums)/sum(wer_denoms)}")
