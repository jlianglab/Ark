from .models import build_model
from .utils import load_pretrained 
from .config import get_config

def create_model(args):
	if args.model_name.lower()  == "vit_base":
		args.cfg = 'simmim/configs/simmim_finetune__vit_base__img224__800ep.yaml'
	elif args.model_name.lower()  == "swin_base":
		args.cfg = 'simmim/configs/simmim_finetune__swin_base__img224_window7__800ep.yaml'
	config = get_config(args)
	model = build_model(config, is_pretrain=False)
	load_pretrained(config, model)
	return model