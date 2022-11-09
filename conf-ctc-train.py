import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  
import wandb
import torch.nn as nn

print("OBJECTIVE: ") #MODIF

wandb_logger = WandbLogger(project="Dysarthric Speech Recognition", log_model='all')

wandb.init(
    project="Dysarthric Speech Recognition", 
    entity="sqrk_", 
    name="ConfCTC-sm-uasp", #MODIF
    tags = ["conformer", "uasp", "ctc", "pre-trained", "control"]) #MODIF

from ruamel.yaml import YAML

main_dir = ""
config_path = main_dir + "Configs/conformer_ctc_bpe_small.yaml" #MODIF

yaml = YAML(typ="safe")
with open(config_path) as f:
    params = yaml.load(f)
print(params)

trainer = pl.Trainer(gpus=1, max_epochs=200, logger=wandb_logger, log_every_n_steps=5)

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small") #MODIF
# model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(main_dir + "Training/Dysarthric Speech Recognition/ConfCTC-sm-uasp-very-low-ckpt (3h5gl3uc)/checkpoints/epoch=56-step=38132.ckpt") # MODIF

#MODIF
# Preserve the decoder parameters in case weight matching can be done later
pretrained_decoder = model.decoder.state_dict()
# model.change_vocabulary(new_tokenizer_dir="/l/users/karima.kadaoui/Research/Training/Tokenizer/UASpeech/tokenizer_spe_bpe_v1024/", new_tokenizer_type="bpe") #MODIF

model.set_trainer(trainer)

params["model"]["train_ds"]["manifest_filepath"] = main_dir + "Manifests/Nemo/UASpeech/with-control/UASpeech-train-manifest.json" #MODIF
params["model"]["validation_ds"]["manifest_filepath"] = main_dir + "Manifests/Nemo/UASpeech/with-control/UASpeech-test-manifest.json" #MODIF
params["model"]["optim"]["lr"] = 1 #MODIF
model.cfg.optim.lr = 1 #MODIF

model._wer.use_cer = False #MODIF

# Insert preserved model weights if shapes match
if model.decoder.decoder_layers[0].weight.shape == pretrained_decoder['decoder_layers.0.weight'].shape:
    model.decoder.load_state_dict(pretrained_decoder)
    print("Decoder shapes matched - restored weights from pre-trained model")
else:
    print("Decoder shapes did not match - could not restore decoder weights from pre-trained model.")

for k,v in params.items(): 
    wandb_logger.experiment.config[k]=v 

model.setup_training_data(train_data_config=params["model"]["train_ds"])
model.setup_validation_data(val_data_config=params["model"]["validation_ds"])

print("CFG:", model.cfg)

wandb.watch(model, log="all", log_graph=(True))

trainer.fit(model)

wandb.finish()
