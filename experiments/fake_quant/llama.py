import argparse
import datautils
import modelutils
import torch
import quik_utils
import quant
import sparseGPT_utils
import os
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def llama_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', type=str,
        help='LLAMA-2 model to load;',
        default='meta-llama/Llama-2-7b-hf', 
        choices=[
        'meta-llama/Llama-2-7b-hf',
        'meta-llama/Llama-2-13b-hf',
        'meta-llama/Llama-2-70b-hf'
        ]
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
    parser.add_argument('--fp_threshold', type=float, default=0.0, help='Threshold where we put the fp features to zero.')
    parser.add_argument('--fp_relative', action='store_true', help='Use relative features for number of fp_features (larger layers have more fp_features)')
    
    # Act. Quantization Params:
    parser.add_argument('--a_bits', type=int, default=16, choices=[4, 8, 16])

    # Weight Quantization Params: 
    parser.add_argument('--w_bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument('--w_clip', action='store_true', help='Use clipping for weight quantization')
    parser.add_argument('--w_asym', action='store_true')
    
    parser.add_argument('--int8_down_proj', action='store_true', help='Use INT8 for Down Projection')
    
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
    parser.add_argument('--hf_token', type=str, default='')
    
    args = parser.parse_args()
    if args.smoothquant or args.int8_2_4:
        assert args.smoothquant and args.int8_2_4 == False, 'Cannot use both SmoothQuant and SparseGPT int8 2:4 quantization!'
    if args.sparsity >0 or args.prunen + args.prunem > 0:
        args.sparseGPT = True
    else:
        args.sparseGPT = False
    return args



@torch.no_grad()
def llama_sequential(model, dataloader, act_scales, dev, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
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
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = modelutils.find_layers(layer)

        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            modules_quik = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)   
                
                # Extract the number of outliers
                if args.fp_relative:
                    outlier_num = int(subset[name].in_features/model.config.hidden_size)*args.fp_features
                else:
                    outlier_num = args.fp_features
                
                layer_scales = None
                if outlier_num > 0:
                    layer_scales = act_scales['model.layers.{}.{}'.format(i, name)]
                    max_val = layer_scales.abs().max()
                    fp_threshold = args.fp_threshold

                    if 'down_proj' in name and args.int8_down_proj:
                        fp_threshold *= 2
                    
                    if max_val <= fp_threshold:
                        outlier_num = 0
                        layer_scales = None
                if args.sparseGPT:
                    modules_quik[name] = sparseGPT_utils.SparseGPT(
                    layer=subset[name],
                    act_scales=layer_scales,
                    fp_features=outlier_num
                    )
                else:
                    modules_quik[name] = quik_utils.QUIK(
                    layer=subset[name],
                    act_scales=layer_scales,
                    fp_features=outlier_num
                    )
                modules_quik[name].quantizer = quant.WeightQuantizer()

                current_w_bits = args.w_bits 
                if 'down_proj' in name:
                    if args.int8_down_proj:
                        current_w_bits = 8
                modules_quik[name].quantizer.configure(
                    current_w_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    modules_quik[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                if args.sparseGPT:
                    modules_quik[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=128)
                else:
                    modules_quik[name].fasterquant(percdamp=args.percdamp, groupsize=-1)
                quantizers['model.layers.%d.%s' % (i, name)] = modules_quik[name].quantizer
                modules_quik[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del modules_quik 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers


if __name__ == '__main__':
    args = llama_parser()
    datautils.set_seed(args.seed)
    
    print(args)
    if args.wandb:
        import wandb
        wandb.init(project="quik", entity=args.wandb_name)
        wandb.config.update(args)
        
    
    model = modelutils.get_llama(args.model, args.hf_token)
    model.eval()
    
    # Extract Scale
    if args.w_bits < 16 or args.a_bits < 16 or args.int8_2_4 or args.smoothquant or args.sparseGPT:
        if args.fp_features > 0 or args.int8_2_4 or args.smoothquant:
            relative_path = os.path.join(modelutils.act_scale_dir, "{}.pt".format(args.model.split('/')[-1]))
            act_scales = torch.load(relative_path)
            print('Loaded act_scales from: ', relative_path)
        else:
            act_scales = None
            print('No act_scales loaded')
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
            synthetic_data=args.synthetic_data, hf_token=args.hf_token
        )
        model.eval()
        sparseGPT_utils.llama_sparsegpt(model, dataloader, DEV, sparseGPT_args)
        import smoothquant_utils
        print('Using SparseGPT int8 2:4 quantization + SmoothQuant Quantization')
        model = smoothquant_utils.quantize_llama(model,
                                               weight_quant='per_channel', act_quant='per_token', 
                                               quantize_bmm_input=False, act_scales=act_scales)
        model.eval()
        args.a_bits = 16
        # del dataloader
        model = model.cpu()
        torch.cuda.empty_cache()    
    elif args.smoothquant:
        import smoothquant_utils
        print('SmoothQuant Quantization')
        model = smoothquant_utils.quantize_llama(model,
                                               weight_quant='per_channel', act_quant='per_token', 
                                               quantize_bmm_input=False, act_scales=act_scales)
        model.eval()
        args.a_bits = 16
        args.w_bits = 16
        model = model.cpu()
        torch.cuda.empty_cache() 
    # Apply GPTQ on the model
    elif args.w_bits < 16 or args.sparseGPT:
        dataloader, testloader = datautils.get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, 
            synthetic_data=args.synthetic_data, hf_token=args.hf_token
        )
        quantizers = llama_sequential(model, dataloader, act_scales, DEV, args)
    
    
    # Add Input Quantization
    if args.a_bits < 16:
        number_of_zero_outlier_linear = 0
        print('.....Activation Quantization.....')
        quant.add_actquant(model)
        layers = modelutils.find_layers(model)

        for name in layers:
            
            # Skip lm_head quantization            
            if 'lm_head' in name:
                print(f'Skipping {name}\n')
                continue 
            
            current_a_bits = args.a_bits
            
            # Extract the number of outliers
            if args.fp_relative:
                outlier_num = int(layers[name].module.in_features/model.config.hidden_size)*args.fp_features
            else:
                outlier_num = args.fp_features
                
            fp_threshold = args.fp_threshold
            if 'down_proj' in name and args.int8_down_proj:
                fp_threshold *= 2
                current_a_bits = 8
            
            if outlier_num > 0 and 'lm_head' not in name:
                max_val = act_scales[name].abs().max()   
                if max_val > fp_threshold:
                    layers[name].fp_features_configure(act_scales[name], outlier_num)
                else:
                    layers[name].fp_features_configure(act_scales[name], 0)
                    number_of_zero_outlier_linear += 1
        

            print(f'{name}: {outlier_num} outliers - {current_a_bits} bits', flush=True)
            layers[name].quantizer.configure(bits=current_a_bits)

        if args.wandb:
            wandb.log({'zero_outlier_linear': number_of_zero_outlier_linear})
        print(f'{number_of_zero_outlier_linear} layers with zero outliers.\n')


    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = datautils.get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token
        )
        print(dataset)
        dataset_ppl = modelutils.llama_eval(model, testloader, DEV)
        print(f'\n{dataset.upper()} PPL: {dataset_ppl:.3f}')
        print(40*'-')
        if args.wandb:
            wandb.log({'ppl/{}'.format(dataset): dataset_ppl})
        