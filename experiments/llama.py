import argparse
import datautils
import modelutils
import torch
import time
import quant_sim
import qlinear
import tqdm
import math


DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def llama_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LLAMA-2 model to load;',
        default='meta-llama/Llama-2-7b-hf',
        choices=[
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-1'
            '3b-hf',
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
    parser.add_argument('--fp_features_num', type=int, default=0, help='Number of features to keep in FP16.')
    parser.add_argument('--fp_features_frac', type=float, default=None, help='Fraction of features to keep in FP16.')

    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length used in evaluations and benchmarks.')

    # Act. Quantization Params:
    parser.add_argument('--a_bits', type=int, default=16, choices=[4, 8, 16])

    # Weight Quantization Params:
    parser.add_argument('--w_bits', type=int, default=16, choices=[4, 8, 16])
    parser.add_argument('--w_asym', action='store_true')

    parser.add_argument('--fp_mlp', action='store_true', help='Use FP16 for MLP')
    parser.add_argument('--fp_attn', action='store_true', help='Use FP16 for Attention Block')
    parser.add_argument('--fp_up_proj', action='store_true', help='Use FP16 for Up Projection')
    parser.add_argument('--fp_down_proj', action='store_true', help='Use FP16 for Down Projection')
    parser.add_argument('--fp_gate_proj', action='store_true', help='Use FP16 for gate Projection')

    parser.add_argument('--int8_down_proj', action='store_true', help='Use INT8 for Down Projection')

    # Wandb Args:
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', type=str, default='')

    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--hf_token', type=str, default='')

    parser.add_argument('--load_qmodel_path', type=str, default=None)
    parser.add_argument('--save_qmodel_path', type=str, default=None)

    parser.add_argument('--sim_eval', action='store_true')
    parser.add_argument('--benchmark', action='store_true')

    args = parser.parse_args()

    if args.int8_down_proj and args.fp_down_proj:
        raise ValueError('Cannot use both INT8 and FP16 for Down Projection!')

    return args


def get_fp_features_num(module: torch.nn.Linear, args):
    fp_features_num = args.fp_features_num
    if args.fp_features_frac is not None:
        fp_features_num = max(int(module.in_features * args.fp_features_frac), fp_features_num)
    return fp_features_num



@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
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
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(len(layers)):
        if i == 0:
            print('Layers: 0', end='', flush=True)
        else:
            print(f', {i}', end='', flush=True)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    torch.cuda.cudart().cudaProfilerStop()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    model.config.use_cache = use_cache

    return ppl.item()


def llama_multigpu(model, gpus):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])
    if hasattr(model.model, 'norm') and model.model.norm is not None:
        model.model.norm = model.model.norm.to(gpus[-1])

    cache = {'mask': None, 'positions': None}

    class MoveModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['positions'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
                cache['positions'] = kwargs['position_ids'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            kwargs['position_ids'] = cache['positions']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


def llama_benchmark(model, testenc, check=False):
    model.config.use_cache = True
    input_ids = testenc.input_ids
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    seq_len = model.seqlen
    nsamples = input_ids.numel() // seq_len
    max_samples = 32
    nsamples = min(nsamples, max_samples)

    cache = {'past': None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = torch.nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        seq_len = model.seqlen
        times = []
        tknps = []
        torch.cuda.cudart().cudaProfilerStart()

        for i in tqdm.tqdm(range(nsamples), desc='Benchmarking', ncols=80):
            batch = input_ids[:, (i * seq_len):((i + 1) * seq_len)].to(DEV)
            start_time = time.perf_counter()
            out = model(
                batch,
                past_key_values=None,
                attention_mask=attention_mask[:, :seq_len].reshape((1, -1))
                #     past_key_values=cache['past'],
                #     attention_mask=attention_mask[:, :(i + 1)*model.seqlen].reshape((1, -1))
            )
            sync()
            times.append(time.perf_counter() - start_time)
            tknps.append(batch.shape[-1] // times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        torch.cuda.cudart().cudaProfilerStop()
        import numpy as np
        print(f'Median times: {np.median(times)} +- {1.96 * np.std(times[2:-2])}')
        print(f'Median tokens/second: {np.median(tknps)} +- {1.96 * np.std(tknps[2:-2])}', )
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
        return np.median(times)


if __name__ == '__main__':
    args = llama_parser()
    datautils.set_seed(args.seed)

    print(args)
    if args.wandb:
        import wandb

        wandb.init(project="quik", entity=args.wandb_name)
        wandb.config.update(args)

    model = modelutils.get_llama(args.model, args.seq_len, args.hf_token)

    # Extract Scale
    if args.w_bits < 16 or args.a_bits < 16:
        if args.fp_features_num > 0 or args.fp_features_frac is not None:
            relative_path = "act_scales/{}.pt".format(args.model.split('/')[-1])
            act_scales = torch.load(relative_path)
        else:
            act_scales = None

    if args.load_qmodel_path:
        print("Load quantized model from ", args.load_qmodel_path)
        save_dict = torch.load(args.load_qmodel_path)
        model.load_state_dict(save_dict["model"])

    if args.a_bits < 16:

        quant_sim.add_actquant(model)
        layers = modelutils.find_layers(model)

        for name in layers:
            # if ".0." not in name:
            #     continue
            # print(f"Found name: {name}")
            bits = args.a_bits
            if 'lm_head' in name or "rotary_emb" in name:
                print(f'Skipping {name}\n')
                continue

            if args.fp_mlp:
                if 'mlp' in name:
                    print(f'Skipping {name}')
                    continue
            if args.fp_up_proj and 'up_proj' in name:
                print(f'Skipping {name}')
                continue

            if 'down_proj' in name:
                if args.fp_down_proj:
                    print(f'Skipping {name}')
                    continue
                elif args.int8_down_proj:
                    bits = 8
                    # print(f'Configuring {name} to INT8')

            if args.fp_down_proj and 'down_proj' in name:
                print(f'Skipping {name}')
                continue
            if args.fp_gate_proj and 'gate_proj' in name:
                print(f'Skipping {name}')
                continue

            if args.fp_attn:
                if 'self_attn' in name:
                    print(f'Skipping {name}')
                    continue
            if args.fp_features_num > 0 or args.fp_features_frac is not None:
                fp_features_num = get_fp_features_num(layers[name].module, args)
                act_name = name
                if "down_proj" in name:
                    fp_features_num = int(args.fp_features_num * layers[name].module.in_features / model.config.hidden_size)
                    fp_features_num = (fp_features_num // 64 + 1) * 64
                layers[name].fp_features_configure(act_scales[act_name], fp_features_num)
            layers[name].quantizer.configure(bits=bits)

    datasets = ['wikitext2']
    if args.sim_eval:
        for dataset in datasets:
            dataloader, testloader = datautils.get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token
            )
            print(dataset)
            dataset_ppl = llama_eval(model, testloader, DEV)
            print(f'\n{dataset.upper()} PPL: {dataset_ppl:.3f}')
            print(40 * '-')
            if args.wandb:
                wandb.log({'ppl/{}'.format(dataset): dataset_ppl})

    if args.benchmark:
        layers = model.model.layers
        # layers = [layers[0]]
        # model.model.layers = torch.nn.ModuleList(layers)
        shared_inputs = {}
        if args.w_bits < 16 and args.a_bits < 16:
            assert not args.w_asym, 'Benchmarking only supports symmetric weight quantization!'
            print("Replace with INT kernels.")
            for i in tqdm.tqdm(range(len(layers)), "Replace with INT4 kernels.", ncols=80):
                opt_block = layers[i]
                sequential = [
                    ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                    ['self_attn.o_proj'],
                    ['mlp.up_proj', 'mlp.gate_proj'],
                    ['mlp.down_proj']
                ]
                full = modelutils.find_layers(opt_block)
                for j, layer_group in enumerate(sequential):
                    subset = {n: full[n] for n in layer_group}
                    shared_inputs[f"{i}.{j}"] = qlinear.SharedQuantizedInput(len(layer_group))
                    for name in subset:
                        start = time.perf_counter()
                        layer = subset[name]
                        if 'lm_head' in name or 'rotary_emb' in name:
                            continue
                        is_quantized = False
                        bits = 16
                        fp_features = 0
                        if isinstance(layer, quant_sim.ActQuantWrapper):
                            if layer.quantizer.configured:
                                is_quantized = True
                                bits = layer.quantizer.bits
                                fp_features = layer.fp_features_num
                            layer = layer.module
                        layer_weight = layer.weight.data

                        layer_scale = torch.ones((layer.out_features, 1))
                        # layer_scale = save_dict['model.layers.{}.{}.scale'.format(i, name)]
                        if fp_features == 0:
                            fp_feature_idx = None
                        else:
                            layer_act_scales = act_scales['model.layers.{}.{}'.format(i, name)].cuda()
                            fp_feature_idx = torch.sort(layer_act_scales)[1][-fp_features:].cpu()

                        if is_quantized:
                            int_mod = qlinear.MixedQLinear.from_float(layer, layer_weight, layer_scale,
                                                                      shared_inputs[f"{i}.{j}"], fp_feature_idx,
                                                                      bits=bits)
                        else:
                            int_mod = layer
                        modelutils.replace_single_mod_opt(opt_block, name, int_mod)

        dataset = 'wikitext2'
        dataloader, testloader = datautils.get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, hf_token=args.hf_token,
        )
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        for i in range(len(gpus)):
            torch.cuda.reset_peak_memory_stats(i)

        llama_benchmark(model, testloader, check=False)
        total_mem_allocated = 0
        total_mem_reserved = 0
        for i in range(len(gpus)):
            total_mem_allocated += torch.cuda.max_memory_allocated(i)
            total_mem_reserved += torch.cuda.max_memory_reserved(i)
        mb = 1024 * 1024
        print(f"Total memory allocated: {total_mem_allocated / mb} Mb, Total memory reserved: {total_mem_reserved / mb}")
