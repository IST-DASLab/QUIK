import torch
import argparse
import datautils
import quant
import modelutils
import quik_utils
import sparseGPT_utils
import os
# from qlinear import *
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def opt_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.',
        default='facebook/opt-1.3b', 
        choices=[
        'facebook/opt-1.3b', 
        'facebook/opt-6.7b', 
        'facebook/opt-13b', 
        'facebook/opt-30b', 
        'facebook/opt-66b']
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.', default='c4'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument('--fp_features', type=int, default=0, help='Number of features to keep in FP16.')
    
    # Act. Quantization Params:
    parser.add_argument('--a_bits', type=int, default=16, choices=[4, 8, 16])

    # Weight Quantization Params: 
    parser.add_argument('--w_bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument('--w_clip', action='store_true', help='Use clipping for weight quantization')
    parser.add_argument('--w_asym', action='store_true')
    
    # SparseGPT arguments:
    parser.add_argument('--sparsity', type=float, default=0, help='Target sparsity')
    parser.add_argument('--prunen', type=int, default=0,help='N for N:M pruning.')
    parser.add_argument('--prunem', type=int, default=0,help='M for N:M pruning.')    
    
    # Wandb Args:
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='anonymous')
    
    parser.add_argument('--int8_2_4', action='store_true', help='Use SparseGPT int8 2:4 quantization with SmoothQuant')
    parser.add_argument('--smoothquant', action='store_true', help='Use SmoothQuant Baseline')
    
    parser.add_argument('--synthetic_data', action='store_true', help='Use Synthetic Data (for debugging purposes)')
    
    args = parser.parse_args()
    if args.smoothquant or args.int8_2_4:
        assert args.smoothquant and args.int8_2_4 == False, 'Cannot use both SmoothQuant and SparseGPT int8 2:4 quantization!'
    if args.sparsity >0 or args.prunen + args.prunem > 0:
        args.sparseGPT = True
    else:
        args.sparseGPT = False
    return args



@torch.no_grad()
def opt_sequential(model, dataloader, act_scales, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        
        print(f'\nLayer: {i}:', end='', flush=True)
        
        layer = layers[i].to(dev)

        subset = modelutils.find_layers(layer)
        modules_quik = {}
        for name in subset:
            
            if args.fp_features > 0:
                layer_scales = act_scales['model.decoder.layers.{}.{}'.format(i, name)]
            else:
                layer_scales = None
            if args.sparseGPT:
                modules_quik[name] = sparseGPT_utils.SparseGPT(
                layer=subset[name],
                act_scales=layer_scales,
                fp_features=args.fp_features
                )
            else:
                modules_quik[name] = quik_utils.QUIK(
                layer=subset[name],
                act_scales=layer_scales,
                fp_features=args.fp_features
                )
            modules_quik[name].quantizer = quant.WeightQuantizer()
            

                
            modules_quik[name].quantizer.configure(
                args.w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )

        def add_batch(name):
            def tmp(_, inp, out):
                modules_quik[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(' {} '.format(name), end='', flush=True)
            if args.sparseGPT:
                modules_quik[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=128)
            else:
                modules_quik[name].fasterquant(percdamp=args.percdamp, groupsize=-1)
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = modules_quik[name].quantizer
            modules_quik[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del modules_quik 
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers


if __name__ == '__main__':

    args = opt_parser()
    datautils.set_seed(args.seed)
    
    print(args)
    if args.wandb:
        import wandb
        wandb.init(project="quik", entity=args.wandb_name)
        wandb.config.update(args)
        
    
    model = modelutils.get_opt(args.model)
    model.eval()

    # Extract Scale
    if args.w_bits < 16 or args.a_bits < 16 or args.int8_2_4 or args.smoothquant or args.sparseGPT:
        if args.fp_features > 0 or args.int8_2_4 or args.smoothquant:
            relative_path = os.path.join(modelutils.act_scale_dir, "{}.pt".format(args.model.split('/')[-1]))
            act_scales = torch.load(relative_path)
            print('Loaded act_scales from: ', relative_path)
        else:
            act_scales = None
            print('No act_scales loaded.')
    
    if args.int8_2_4:
        
        class SparseGPTArgs:
            def __init__(self):
                self.seed = args.seed
                self.nsamples = args.nsamples
                self.model = args.model
                self.dataset = args.dataset
                self.synthetic_data = args.synthetic_data
                self.wandb = args.wandb
                self.wandb_name = args.wandb_name
                self.prunen = 2
                self.prunem = 4
                self.blocksize = 128
                self.gmp = False
                self.wbits = 16
                self.minlayer = -1
                self.maxlayer = 1000
                self.prune_only = ''
                self.invert = False
                self.save = ''
                self.log_wandb = False
                self.sparsity = 0.0
                self.percdamp = args.percdamp
                
        sparseGPT_args = SparseGPTArgs()
        print('Using SparseGPT int8 2:4 quantization with args:\n', sparseGPT_args.__dict__)
        dataloader, testloader = datautils.get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, 
            synthetic_data=args.synthetic_data
        )
        model.eval()
        sparseGPT_utils.opt_sparsegpt(model, dataloader, DEV, sparseGPT_args)
        
        import smoothquant_utils
        print('Using SparseGPT int8 2:4 quantization + SmoothQuant Quantization')
        model = smoothquant_utils.quantize_opt(model,
                                               weight_quant='per_channel', act_quant='per_token', 
                                               quantize_bmm_input=False, act_scales=act_scales)
        model.eval()
        args.a_bits = 16
        del dataloader
        model = model.cpu()
        torch.cuda.empty_cache()    
    elif args.smoothquant:
        import smoothquant_utils
        print('SmoothQuant Quantization')
        model = smoothquant_utils.quantize_opt(model,
                                               weight_quant='per_channel', act_quant='per_token', 
                                               quantize_bmm_input=False, act_scales=act_scales)
        model.eval()
        args.a_bits = 16
        args.w_bits = 16
        model = model.cpu()
        torch.cuda.empty_cache()   
        
    elif args.w_bits < 16 or args.sparseGPT:
        dataloader, testloader = datautils.get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, 
            synthetic_data=args.synthetic_data
        )
        quantizers = opt_sequential(model, dataloader, act_scales, DEV, args)
    
        
    if args.a_bits < 16:

        quant.add_actquant(model)
        layers = modelutils.find_layers(model)

        for name in layers:
            
            if args.fp_features > 0:
                layers[name].fp_features_configure(act_scales[name], args.fp_features)
            
            # Skip lm_head quantization  
            if 'lm_head' in name:
                print(f'\nSkipping {name}\n')
                continue 
            
                
            layers[name].quantizer.configure(bits=args.a_bits)


    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = datautils.get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        dataset_ppl = modelutils.opt_eval(model, testloader, DEV)
        print(f'\n{dataset.upper()} PPL: {dataset_ppl:.3f}')
        print(40*'-')
        if args.wandb:
            wandb.log({'ppl/{}'.format(dataset): dataset_ppl})
        