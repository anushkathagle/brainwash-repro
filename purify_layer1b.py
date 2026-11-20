"""
purify_layer1b.py

Layer 1b defenses against BrainWash: PureVQ-GAN (discrete VQ bottleneck) and PureGen-EBM (mid-run
Langevin dynamics of an energy-based model) purification of the poisoned task's training images,
applied before continual learning proceeds. DiffPure is Layer 1a (purify_*_layer1a.py).

Same design rule as the Layer 1a scripts: this is a deliberate fork of main_baselines.py, NOT an edit
of it. main_baselines.py, approaches/*.py and repro/sweep/* are untouched, so the original
reproduction stays byte-for-byte reproducible. Every line that is not part of the original
main_baselines.py is marked "# >>> LAYER1B" / "# <<< LAYER1B". Defense flags are parsed here, in
__main__, and stripped from sys.argv before the original get_args() sees it, so approaches/arguments.py
never has to change.

Arms (selected purely by CLI flags; the first three are identical to main_baselines.py):
    Arm A (clean, no defense):     --checkpoint <noise.pkl>
    Arm B (poison, no defense):    --checkpoint <noise.pkl> --addnoise
    Arm B'(uniform, no defense):   --checkpoint <noise.pkl> --addnoise --uniform
    PureVQ-GAN, poison:            --checkpoint <noise.pkl> --addnoise --purifier <purifier.pt> [--purify_passes 1]
    PureVQ-GAN, clean (utility):   --checkpoint <noise.pkl> --purifier <purifier.pt>
    PureGen-EBM, poison:           --checkpoint <noise.pkl> --addnoise --ebm_purify --ebm_steps 100 --ebm_eps 0.025 \
                                       --puregen_repo <clone of SunayBhat1/PureGen_PoisonDefense>
    PureGen-EBM, clean (utility):  --checkpoint <noise.pkl> --ebm_purify --ebm_steps 100 --ebm_eps 0.025 --puregen_repo ...
The clean-with-defense arms are the matched baseline: purification alone changes forgetting (it acts as a
regularizer), so recovery is best reported against clean+same-defense as well as against plain clean.

Requirements beyond the base brainwash-repro environment (see repro/defense/README_*.md):
    - PureVQ-GAN: the purevqgan/ package in this repo and a trained purifier.pt (python -m purevqgan.train).
    - PureGen-EBM: the puregen_ebm/ package in this repo, `pip install huggingface_hub safetensors`, and a clone of
      https://github.com/SunayBhat1/PureGen_PoisonDefense (its models/EBMs.py is imported at run time, not copied).
"""
import sys, os, time
import numpy as np
import pickle as pkl    
import utils
import torch
from approaches.arguments import get_args
from resnet import ResNet18

# >>> LAYER1B: stdlib import for the short output names below
import hashlib
# <<< LAYER1B


tstart = time.time()

# >>> LAYER1B: defaults + the two purification helpers. Both take (args, xtrain, xtrain_clean): xtrain is the
# poisoned (or uniform-noised, or clean) task-T training set exactly as the original code builds it, xtrain_clean
# the same images in the same order WITHOUT noise, used only to report how much perturbation survives.
DEFAULT_OUT_DIR = 'repro/sweep/purevq_results/layer1b_out'


def _report_str(rep):
    return ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in rep.items())


def purify_task_data_pvq(args, xtrain, xtrain_clean):
    """Layer 1b (PureVQ-GAN): one forward pass (or --purify_passes) through a trained VQ-GAN."""
    from purevqgan import load_purifier, purify_tensor
    from purevqgan.purify import perturbation_report
    G = load_purifier(args.purifier, device='cuda')
    if G.config['img_size'] != xtrain.shape[-1]:
        raise SystemExit(f"[LAYER1B] purifier {args.purifier} was trained at {G.config['img_size']}x{G.config['img_size']} but "
                         f"the task images are {xtrain.shape[-1]}x{xtrain.shape[-1]}; train one at this resolution")
    print(f'[LAYER1B] PureVQ-GAN purifying with {args.purifier} (passes={args.purify_passes}, config={G.config})')
    t0 = time.time()
    x_pure = purify_tensor(G, xtrain, passes=args.purify_passes)
    print(f'[LAYER1B] purified {len(xtrain)} samples in {time.time() - t0:.1f}s')
    print('[LAYER1B] Purification report : ' + _report_str(perturbation_report(G, xtrain_clean, xtrain, passes=args.purify_passes)))
    del G
    torch.cuda.empty_cache()
    return x_pure.to(xtrain.device, xtrain.dtype)


def purify_task_data_ebm(args, xtrain, xtrain_clean):
    """Layer 1b (PureGen-EBM): mid-run Langevin dynamics of the authors' pretrained EBM (paper Eq. 9)."""
    from puregen_ebm import load_ebm, purify_with_report
    ebm = load_ebm(args.ebm_path, args.puregen_repo, device='cuda')
    print(f'[LAYER1B] PureGen-EBM purifying with {args.ebm_path} (steps={args.ebm_steps}, temp={args.ebm_temp}, '
          f'eps={args.ebm_eps}, init_noise={args.ebm_init_noise})')
    t0 = time.time()
    x_pure, rep = purify_with_report(ebm, xtrain_clean, xtrain, args.ebm_steps, temp=args.ebm_temp,
                                     eps=args.ebm_eps, init_noise=args.ebm_init_noise)
    print(f'[LAYER1B] purified {len(xtrain)} samples in {time.time() - t0:.1f}s')
    print('[LAYER1B] Purification report : ' + _report_str(rep))
    del ebm
    torch.cuda.empty_cache()
    return x_pure.to(xtrain.device, xtrain.dtype)


def defense_tag(args):
    """Short suffix naming the defense arm (used in acc_mat_*.npy / summary.tsv)."""
    if args.purifier is not None:
        tag = args.purify_tag or os.path.basename(os.path.dirname(os.path.abspath(args.purifier)))
        return f'pvq-{tag}_p{args.purify_passes}'
    if args.ebm_purify:
        return f'ebm-s{args.ebm_steps}_e{args.ebm_eps}' + (f'_i{args.ebm_init_noise}' if args.ebm_init_noise else '')
    return None
# <<< LAYER1B

def main(args):

    if args.checkpoint != None:
        checkpoint_dict = pkl.load(open(args.checkpoint, 'rb'))   

    if args.approach == 'afec_ewc' or args.approach == 'ewc' or args.approach == 'afec_rwalk' or args.approach == 'rwalk' or args.approach == 'afec_mas' or args.approach == 'mas' or args.approach == 'afec_si' or args.approach == 'si' or args.approach == 'ft' or args.approach == 'random_init' or args.approach == 'rwalk2':
        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed,
                                                                        args.lamb, args.lr, args.batch_size, args.nepochs)
    elif args.approach == 'gs':
        log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment,
                                                                                            args.approach, args.seed, 
                                                                                            args.lamb, args.mu, args.rho,
                                                                                                    args.eta,
                                                                                            args.lr, args.batch_size, args.nepochs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print('[CUDA unavailable]'); sys.exit()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Args -- Experiment
    if args.experiment == 'split_cifar100':
        # from dataloaders import split_cifar100 as dataloader
        from approaches.data_utils import generate_split_cifar100_tasks
    elif args.experiment == 'split_mini_imagenet':
        from approaches.data_utils import generate_split_mini_imagenet_tasks
    elif args.experiment == 'split_tiny_imagenet':
        from approaches.data_utils import generate_split_tiny_imagenet_tasks


    # Args -- Approach

    if args.approach == 'afec_ewc':
        from approaches import afec_ewc as approach
    elif args.approach == 'ewc':
        from approaches import ewc as approach
    elif args.approach == 'rwalk':
        from approaches import rwalk as approach
    elif args.approach == 'mas':
        from approaches import mas as approach


    print('Load data...')

    if args.experiment == 'split_cifar100':
        order = np.arange(100)
        im_sz = 32
        emb_fact = 1    
        data, taskcla, inputsize, task_order = generate_split_cifar100_tasks(args.tasknum, args.seed, rnd_order=False, order=order)
    elif args.experiment == 'split_mini_imagenet':
        order = np.arange(100)  
            
        class_num = 100 // (args.tasknum)  
        im_sz = 84
        emb_fact = 1    

        order = np.arange(100)
        home = os.path.expanduser('~')
        mini_root = os.path.join(home, 'data', 'miniImagenet' ) 
        data, taskcla, inputsize, task_order = generate_split_mini_imagenet_tasks(mini_root, task_num = args.tasknum, 
                                                                    rnd_order=False, order=order) 
        
    elif args.experiment == 'split_tiny_imagenet':
        order = np.arange(200)  
        home = os.path.expanduser('~')  
        root_add = os.path.join(home, 'data', 'tiny-imagenet-200') 
        dataset_file = './data/tiny_imagenet.npz'
        data, taskcla, inputsize, task_order = generate_split_tiny_imagenet_tasks(task_num = args.tasknum, 
                                                                    rnd_order=False, save_data=False,
                                                                    dataset_file=dataset_file, 
                                                                    order=order, root_add=root_add)
        
        class_num = 200 // (args.tasknum)  
        im_sz = 64
        emb_fact = 9
        
    print('\nInput size =', inputsize, '\nTask info =', taskcla)


    ########################################################################################################################

    print('Inits...')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    nf = 32

    net = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()
    net_emp = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()


    ########################################################################################################################

    save_dict = {}  
    save_dict['scenario'] = args.scenario_name
    save_dict['model_type'] = 'resnet'    
    save_dict['dataset'] = args.experiment
    save_dict['class_num'] = data['ncla'] // args.tasknum  
    save_dict['bs'] = args.batch_size
    save_dict['lr'] = args.lr
    save_dict['n_epochs'] = args.nepochs
    save_dict['model'] = net.state_dict()
    save_dict['model_name'] = net.__class__.__name__
    save_dict['task_num'] = args.lasttask        
    save_dict['task_order'] = task_order
    save_dict['seed'] = args.seed    
    save_dict['emb_fact'] = emb_fact  
    save_dict['im_sz'] = inputsize[1]  

    cont_method_args = {'method': args.approach} 
    for tmp_key in args.__dict__.keys():    
        cont_method_args[tmp_key] = args.__dict__[tmp_key] 

    save_dict['cont_method_args'] = cont_method_args    

    if 'afec' in args.approach:
        if args.checkpoint is not None:
            lamb = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb']  
            lamb_emp = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb_emp']  
        else:
            lamb = args.lamb    
            lamb_emp = args.lamb_emp    

        appr = approach.Appr(net, sbatch=args.batch_size, lamb=lamb, lamb_emp=lamb_emp, 
                            lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name,
                            empty_net = net_emp, clipgrad=args.clip)
    else:
        if args.checkpoint is not None:
            lamb = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb']  
        else:
            lamb = args.lamb    

        appr = approach.Appr(net, lamb=lamb, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, clipgrad=args.clip)


    if args.checkpoint is not None:
        appr.load_model(checkpoint_dict['pretrained_ckpt']['model'])   
        if 'afec' in args.approach:
            appr.load_emp_model(checkpoint_dict['pretrained_ckpt']['cont_method_args']['model_emp'])

        if args.init_acc:
            accs_tmp = []
            for u in range(checkpoint_dict['pretrained_ckpt']['task_num']):  
                xtest = data[u]['test']['x']
                ytest = data[u]['test']['y']
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                accs_tmp.append(test_acc *100)   
            
            with np.printoptions(precision=2, suppress=True):   
                print(np.array(accs_tmp) ) 
            
        

    print(appr.criterion)
    print('-' * 100)
    relevance_set = {}

    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

    for t, ncla in taskcla:
        if args.checkpoint is not None and t < args.lasttask:
            print('Skip task {:2d} : {:15s}'.format(t, data[t]['name']))
            continue


        if t==1 and 'find_mu' in args.date:
            break

        if t == args.lasttask and args.checkpoint is None:  
            break
        
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        # Get data
        xtrain = data[t]['train']['x'].clone()
        xvalid = data[t]['test']['x'].clone()   

        ytrain = data[t]['train']['y'].clone()
        yvalid = data[t]['test']['y'].clone()

        # >>> LAYER1B: the same task images without noise (replaced below if noise is added)
        xtrain_clean = xtrain
        # <<< LAYER1B

        if args.checkpoint is not None and args.addnoise == True:
            
            if args.uniform is True:
                print('Using uniform noise')
                if 'inj_data_idx' not in checkpoint_dict.keys():
                    all_noise = torch.rand_like(xtrain) * 2 * checkpoint_dict['delta'] - checkpoint_dict['delta']
                else:
                    print('Using uniform noise only on the injected data')
                    noise_prm = checkpoint_dict['rnd_idx_train']   
                    xtrain = xtrain[noise_prm]  
                    ytrain = ytrain[noise_prm]
                    all_noise = torch.zeros_like(xtrain)    
                    inj_idx = checkpoint_dict['inj_data_idx']   
                    print(f'number of noisy data : {len(inj_idx)}')
                    all_noise[inj_idx] = torch.rand_like(xtrain[inj_idx]) * 2 * checkpoint_dict['delta'] - checkpoint_dict['delta']

                # >>> LAYER1B
                xtrain_clean = xtrain
                # <<< LAYER1B
                xtrain = torch.clamp(xtrain + all_noise, 0, 1)  

                
            else:
                print('Using noise from checkpoint')
                all_noise = checkpoint_dict['latest_noise']   
                noise_prm = checkpoint_dict['rnd_idx_train']   
                xtrain = xtrain[noise_prm]
                ytrain = ytrain[noise_prm]

            
                # >>> LAYER1B
                xtrain_clean = xtrain
                # <<< LAYER1B
                xtrain = torch.clamp(xtrain + all_noise, 0, 1)  
        
        # >>> LAYER1B: purify xtrain (poisoned, uniform or clean, whichever arm this is). Sits AFTER the
        # rnd_idx_train permutation and the noise addition, so it is a pure input transform.
        assert not (args.purifier is not None and args.ebm_purify), "choose one of --purifier / --ebm_purify"
        if args.purifier is not None:
            xtrain = purify_task_data_pvq(args, xtrain, xtrain_clean)
        elif args.ebm_purify:
            xtrain = purify_task_data_ebm(args, xtrain, xtrain_clean)
        # <<< LAYER1B

        task = t

        # Train
        appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
        print('-' * 100)

        # Test
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                        100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss
            
        # Save
        
        print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))
        print('Save at ' + args.output)

        with np.printoptions(precision=2, suppress=True):   
            print(acc)

    if args.checkpoint is not None: 
        acc[:args.lasttask, :args.lasttask] = checkpoint_dict['pretrained_ckpt']['acc_mat'][:args.lasttask, :args.lasttask]
        
    # Done
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    if args.checkpoint is not None: 
        acc_mat_sace_name = args.checkpoint.split('/')[-1]
        #remove the .pkl extension
        acc_mat_sace_name = acc_mat_sace_name[:-4]

        if args.addnoise is False:
            method = 'clean'
        if args.addnoise and args.uniform:  
            method = 'uniform'
        elif args.addnoise and args.uniform is False:
            method = 'ours'    


        # >>> LAYER1B: REPLACES the original `np.save(f'acc_mat_{name}_{method}.npy', acc)` (repo root, long names).
        # Short, collision-free names and an --out_dir, so defended arms never overwrite the baseline and
        # long AFEC pkl names cannot trigger ENAMETOOLONG after a multi-hour run (same fix as the Layer 1a scripts).
        acc_mat_sace_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        if not acc_mat_sace_name or acc_mat_sace_name in ('.', '/'):
            acc_mat_sace_name = os.path.basename(args.checkpoint)[:-4]
        if len(acc_mat_sace_name) > 100:
            acc_mat_sace_name = acc_mat_sace_name[:80] + "_" + hashlib.sha1(acc_mat_sace_name.encode()).hexdigest()[:10]
        _dtag = defense_tag(args)
        if _dtag is not None:
            method = f"{method}_{_dtag}"
        os.makedirs(args.out_dir, exist_ok=True)
        out_npy = os.path.join(args.out_dir, f'acc_mat_{acc_mat_sace_name}_{method}.npy')
        try:
            np.save(out_npy, acc)
            print(f'[LAYER1B] saved accuracy matrix -> {out_npy}')
        except OSError as e:
            print(f'[LAYER1B] WARNING: could not save accuracy matrix ({e}); continuing so After BWT is not lost')
        # <<< LAYER1B




    bwt_before = np.mean((acc[args.lasttask-1] - np.diag(acc))[:args.lasttask-1][:-1])
    avg_acc_before  = np.mean(acc[args.lasttask-1, :args.lasttask])

    if args.checkpoint is not None: 
        bwt_after = np.mean((acc[-1] - np.diag(acc))[:-1])
        avg_acc_after  = np.mean(acc[-1][:-1])
        last_task_acc = acc[-1, -1] 
        print(f'After BWT : {bwt_after} After avg acc : {avg_acc_after} Last task acc : {last_task_acc}')   

        # >>> LAYER1B: one machine-readable row per run, next to the .npy
        summary = os.path.join(args.out_dir, 'summary.tsv')
        _new = not os.path.isfile(summary)
        with open(summary, 'a') as fh:
            if _new:
                fh.write('method\tapproach\texperiment\tseed\tdefense\tbwt_after\tavg_acc_after\tlast_task_acc\tnoise_pkl\n')
            fh.write(f'{method}\t{args.approach}\t{args.experiment}\t{args.seed}\t{defense_tag(args) or "-"}\t'
                     f'{bwt_after:.4f}\t{avg_acc_after:.4f}\t{last_task_acc:.4f}\t{os.path.basename(args.checkpoint)}\n')
        print(f'[LAYER1B] appended summary row -> {summary}')
        # <<< LAYER1B

        
    print(f'Before BWT : {bwt_before} Before avg acc : {avg_acc_before}')  

    save_dict['last_task'] = int(args.lasttask)
    save_dict['acc_mat'] = acc
    save_dict['avg_acc'] = np.mean(acc[-1, :args.lasttask])
    save_dict['bwt'] = bwt_before
    save_dict['model'] = net.state_dict()
    if 'afec' in args.approach: 
        save_dict['cont_method_args']['model_emp'] = net_emp.state_dict()   

    # save_dict['optim'] = optim.state_dict()

    save_name = utils.generate_save_name(save_dict)
    if args.checkpoint is None:
        #check if the file exists and add a number to the end if it does    
        if os.path.exists(f'{args.approach}_{save_name}.pkl'):
            print(f'File {args.approach}_{save_name}.pkl already exists. Saving with a different name.')
            if 'afec' not in args.approach:
                pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_{save_name}_1.pkl', 'wb'))
            else:
                pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_{save_name}_1.pkl', 'wb'))

        else:
            if 'afec' not in args.approach:
                pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_{save_name}.pkl', 'wb'))
            else:
                pkl.dump(save_dict, open(f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_{save_name}.pkl', 'wb'))


if __name__ == '__main__':
    # >>> LAYER1B: parse the defense flags ourselves so approaches/arguments.py never has to be touched; strip
    # them from sys.argv before handing the rest to the original get_args().
    import argparse
    _dp = argparse.ArgumentParser(add_help=False)
    _dp.add_argument('--purifier', default=None, type=str, help='PureVQ-GAN: path to a trained purifier.pt')
    _dp.add_argument('--purify_passes', default=1, type=int, help='PureVQ-GAN: refinement passes x <- G(x)')
    _dp.add_argument('--purify_tag', default=None, type=str, help='name for output files (default: purifier folder name)')
    _dp.add_argument('--ebm_purify', action='store_true', help='PureGen-EBM purification of the task-T training data')
    _dp.add_argument('--ebm_path', default='SunayBhat1/puregen-ebm-cinic10-imagenet', type=str, help='HF id or local dir')
    _dp.add_argument('--puregen_repo', default=None, type=str, help='clone of SunayBhat1/PureGen_PoisonDefense')
    _dp.add_argument('--ebm_steps', default=150, type=int, help='Langevin steps (authors: 150)')
    _dp.add_argument('--ebm_temp', default=1e-4, type=float, help='Langevin temperature (authors: 1e-4)')
    _dp.add_argument('--ebm_eps', default=1.25e-2, type=float, help='Langevin step size (authors: 1.25e-2)')
    _dp.add_argument('--ebm_init_noise', default=0.0, type=float, help='Gaussian noise std ([-1,1] units) before the chain')
    _dp.add_argument('--out_dir', default=DEFAULT_OUT_DIR, type=str, help='where to write acc_mat_*.npy and summary.tsv')
    _defense_args, _remaining_argv = _dp.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining_argv
    # <<< LAYER1B

    args = get_args()

    # >>> LAYER1B
    for _k, _v in vars(_defense_args).items():
        setattr(args, _k, _v)
    # <<< LAYER1B
    main(args)
    