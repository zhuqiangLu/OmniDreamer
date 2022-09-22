import importlib
import yaml
import os
import numpy as np
import cv2
import torch
from omegaconf import OmegaConf
from PIL import Image
import cv2
import argparse 
import glob
import tqdm

def show_image(s, save_path):
    s = s.detach().cpu().numpy().transpose(0,2,3,1)[0]
    s = ((s+1.0)*127.5).clip(0,255).astype(np.uint8)
    if s.shape[2] == 1:
        s = np.tile(s, (1,1,3))
    elif s.shape[2] > 3:
        #print("condition_img is rounded to 3 ch")
        s = s[:,:,:3]
    s = Image.fromarray(s)
    s.save(save_path)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)




if __name__ == "__main__":
    config_path = './configs/sun360_basic_transformer.yaml'
    ckpt_path = '/share/zhlu6105/checkpoints/transformer.ckpt'
    outdir = '/share/zhlu6105/omnidreamer/exp_out'
    os.makedirs(outdir, exist_ok=True)

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location='cpu')['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.cuda().eval()

    torch.set_grad_enabled(False)

    data = instantiate_from_config(config.data)
    data.setup()
    dataloader = data._test_dataloader()

    for num_1, batch in enumerate(dataloader):
        h, w = 256, 256
        concat_inputs = batch['concat_input'].permute(0,3,1,2).float().cuda()
        x = batch['image'].permute(0,3,1,2).float().cuda()
        masked_xs = batch['masked_image'].permute(0, 3, 1,2).float().cuda()
        for num_2, cond_input in enumerate(concat_inputs):
            num = (num_1+1) * (num_2 + 1)
            cond_input = cond_input.unsqueeze(dim=0)
            x_raw = x[num_2].unsqueeze(dim=0)
            masked_x = masked_xs[num_2].unsqueeze(dim=0)
            c_code, c_indices = model.encode_to_c(cond_input)
    
            codebook_size = config.model.params.first_stage_config.params.embed_dim
            z_indices_shape = c_indices.shape 
            z_code_shape = c_code.shape 
            z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)
            x_sample = model.decode_to_img(z_indices, z_code_shape)
            print("Running the first stage (completion)")
    
            import time
    
            idx = z_indices
            idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])
    
            cidx = c_indices
            cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])
            
    
            temperature = 1.0
            top_k = None
            update_every = 50
    
            start_t = time.time()
    
            ## Please change for circular inference of transformer.
            circular_type = 0 # if 0, no circular inference, just raster scan
            px = 8 # Adjust according an input resion.  eg. 90deg:4 180deg:8
            ##
    
    
            # if circular_type == 1:
            #     idx = torch.cat((idx[:,:,-px:], idx, idx[:,:,:px]), 2)
            #     z_code_shape = [idx.shape[0], 256, idx.shape[1], idx.shape[2]]
            #     cidx = torch.cat((cidx[:,:,-px:], cidx, cidx[:,:,:px]), 2)
            #     c_code_shape = [cidx.shape[0], 256, cidx.shape[1], cidx.shape[2]]
    
            for i in range(0, z_code_shape[2]-0):
                if i <= 8:
                    local_i = i
                elif z_code_shape[2]-i < 8:
                    local_i = 16-(z_code_shape[2]-i)
                else:
                    local_i = 8
                for j in range(0,z_code_shape[3]-0):
                    if j <= 16:
                        local_j = j
                    elif z_code_shape[3]-j < 16:
                        local_j = 16-(z_code_shape[3]-j)
                    else:
                        local_j = 16
    
                    i_start = i-local_i
                    i_end = i_start+16
                    j_start = j-local_j
                    j_end = j_start+16
                    
                    patch = idx[:,i_start:i_end,j_start:j_end]
                    patch = patch.reshape(patch.shape[0],-1)
                    cpatch = cidx[:, i_start:i_end, j_start:j_end]
                    cpatch = cpatch.reshape(cpatch.shape[0], -1)
                    patch = torch.cat((cpatch, patch), dim=1)
                    logits,_ = model.transformer(patch[:,:-1])
                    logits = logits[:, 255:, :]
                    logits = logits.reshape(z_code_shape[0],16,16,-1)
                    logits = logits[:,local_i,local_j,:]
    
                    logits = logits/temperature
    
                    if top_k is not None:
                        logits = model.top_k_logits(logits, top_k)
    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    idx[:,i,j] = torch.multinomial(probs, num_samples=1)
    
                    step = i*z_code_shape[3]+j
                    if step%update_every==0 or step==z_code_shape[2]*z_code_shape[3]-1:
                        pass
                    
            # copy to other side
            if circular_type ==1:
                idx[:,:,:px] = idx[:,:,-2*px:-px]
                idx[:,:,-px:] = idx[:,:,px:2*px] 
    
            # idx = torch.cat((idx[:,:,16:], idx[:,:,:16]),2) #lr_replace
            # patch = idx.reshape(idx.shape[0], -1)
            # cpatch = cidx.reshape(cidx.shape[0], -1)
            # transformer_input = torch.cat((cpatch, patch), dim=1)
            # logits, _ = model.transformer(transformer_input[:, :-1])
            # logits = logits[:, cpatch.shape[1]-1:]
            # print(logits.shape)
            # probs = torch.nn.functional.softmax(logits, dim=-1)
            # idx = torch.argmax(probs, dim=-1)
            # print(logits.shape, idx.shape)

    
            x_sample = model.decode_to_img(idx, z_code_shape)
            #print(f"Time: {time.time() - start_t} seconds")
            #print(f"Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
            #show_image(x_sample, os.path.join(args.outdir, "sample_%08d.png" % num))
    
            # drop both edges
            if circular_type == 1:
                idx = idx[:,:,px:-px]
                z_code_shape = [z_code_shape[0], z_code_shape[1], z_code_shape[2], idx.shape[2]]
                x_sample = x_sample[:,:,:,16*px:-16*px]
            # x_sample = torch.cat((x_sample[:,:,:,x_sample.shape[3]//2:], x_sample[:,:,:,:x_sample.shape[3]//2]), 3)
            show_image(x_sample, os.path.join(outdir, "sample_%08d.png" % (num)))
            show_image(x_raw, os.path.join(outdir, "x_%08d.png" % (num)))
            show_image(masked_x, os.path.join(outdir, 'mask_%08d.png' % (num)))
    
        break




