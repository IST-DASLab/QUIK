import torch
import argparse
import datautils
import quant
import modelutils
import quik_utils
import sparseGPT_utils
import os

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def falcon_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model', type=str,
        help='Falcon model to load; pass `tiiuae/falcon-X`.',
        default='tiiuae/falcon-7b', 
        choices=[
            'tiiuae/falcon-7b',
            'tiiuae/falcon-40b',
            'tiiuae/falcon-180B',
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
        '--percdamp', type=float, default=0.5,
        help='Percent of the average Hessian diagonal to use for dampening (default: 0.1 for Falcon models).'
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
        
    parser.add_argument('--int8_fc2', action='store_true', help='Use INT8 for FC2')
    
    # SparseGPT arguments:
    parser.add_argument('--sparsity', type=float, default=0, help='Target sparsity')
    parser.add_argument('--prunen', type=int, default=0,help='N for N:M pruning.')
    parser.add_argument('--prunem', type=int, default=0,help='M for N:M pruning.')  
    parser.add_argument('--prune_only', type=str, default='',help='Prune only layers that contain this text.')  
    parser.add_argument('--prune_only2', type=str, default='',help='Prune only layers that contain this text.')  
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
def falcon_sequential(model, dataloader, act_scales, dev, args):

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h
    
    
    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, 'alibi': None}
    
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    
    layers[0] = Catcher(layers[0])    
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']
    print('Ready.')
    
    quantizers = {}
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = modelutils.find_layers(layer)
        sequential = [
            ["self_attention.query_key_value"],
            ["self_attention.dense"],
            ["mlp.dense_h_to_4h"],
            ["mlp.dense_4h_to_h"],
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
                    layer_scales = act_scales['transformer.h.{}.{}'.format(i, name)]
                    max_val = layer_scales.abs().max()
                    fp_threshold = args.fp_threshold
                    
                    if 'dense_4h_to_h' in name and args.int8_fc2:
                        fp_threshold *= 2
                    
                    if max_val <= fp_threshold:
                        outlier_num = 0
                        layer_scales = None

                if args.sparseGPT and ((args.prune_only in name or args.prune_only == '') or (args.prune_only2 in name or args.prune_only2 == '')):
                    print(f'Prune {name}')
                    modules_quik[name] = sparseGPT_utils.SparseGPT(
                    layer=subset[name],
                    act_scales=layer_scales,
                    fp_features=outlier_num
                    )
                else:
                    print(f'Dense {name}')
                    modules_quik[name] = quik_utils.QUIK(
                    layer=subset[name],
                    act_scales=layer_scales,
                    fp_features=outlier_num
                    )
                modules_quik[name].quantizer = quant.WeightQuantizer()
                current_w_bits = args.w_bits 
                if 'dense_4h_to_h' in name and args.int8_fc2:
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
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            for h in handles:
                h.remove()

            for name in subset:
                if args.sparseGPT and ((args.prune_only in name or args.prune_only == '') or (args.prune_only2 in name or args.prune_only2 == '')):
                    modules_quik[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=128)
                else:
                    modules_quik[name].fasterquant(percdamp=args.percdamp, groupsize=-1)
                quantizers['transformer.h.%d.%s' % (i, name)] = modules_quik[name].quantizer
                modules_quik[name].free()
    
    
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        for param in layers[i].parameters(): 
            is_nan = torch.isnan(param).sum()
            if is_nan > 0:
                print('NaN in layer {}'.format(i))
                raise ValueError
        del layer
        del modules_quik 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    model.config.use_cache = use_cache
    return quantizers

  
if __name__ == '__main__':
    
    args = falcon_parser()
    datautils.set_seed(args.seed)
    
    print(args)
    if args.wandb:
        import wandb
        wandb.init(project="quik", entity=args.wandb_name)
        wandb.config.update(args)
        
    
    model = modelutils.get_falcon(args.model, args.hf_token)
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
        raise NotImplementedError
    elif args.smoothquant:
        import smoothquant_utils
        print('SmoothQuant Quantization')
        model = smoothquant_utils.quantize_falcon(model,
                                               weight_quant='per_channel', act_quant='per_token', 
                                               quantize_bmm_input=False, act_scales=act_scales)
    elif args.w_bits < 16 or args.sparseGPT:
        dataloader, _ = datautils.get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, 
                synthetic_data=args.synthetic_data, hf_token=args.hf_token
            )
        quantizers = falcon_sequential(model, dataloader, act_scales, DEV, args)
        del quantizers
        
    
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
            if 'dense_4h_to_h' in name and args.int8_fc2:
                fp_threshold *= 2
            
            if outlier_num > 0 and 'lm_head' not in name:
                max_val = act_scales[name].abs().max()

                if max_val > fp_threshold:
                    layers[name].fp_features_configure(act_scales[name], outlier_num)
                else:
                    outlier_num = 0
                    layers[name].fp_features_configure(act_scales[name], 0)
                    number_of_zero_outlier_linear += 1
            
            # Check if FC2 is INT8
            if 'dense_4h_to_h' in name and args.int8_fc2:
                current_a_bits = 8
            
            print(f'{name}: {outlier_num} outliers - {current_a_bits} bits', flush=True)
            layers[name].quantizer.configure(bits=current_a_bits)
            
        if args.wandb:
            wandb.log({'zero_outlier_linear': number_of_zero_outlier_linear})
        print(f'{number_of_zero_outlier_linear} layers with zero outliers.\n')

    datasets = ['wikitext2'] 
    for dataset in datasets:
        dataloader, testloader = datautils.get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        del dataloader
        print(dataset)
        dataset_ppl = modelutils.falcon_eval(model, testloader, DEV)
        print(f'\n{dataset.upper()} PPL: {dataset_ppl:.3f}')
        print(40*'-')
        if args.wandb:
            wandb.log({'ppl/{}'.format(dataset): dataset_ppl})
        


