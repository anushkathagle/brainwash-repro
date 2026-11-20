"""
friends_layer2_p1.py  --  COPY of friends_layer2.py that uses P1 noise (friends_p1.py) instead of FRIENDS' friendly
noise: eta keeps head T's prediction but disturbs the shared features. Extra flags: --p1_beta (weight of "disturb
the features", default 1) and --p1_lambda (weight of "make eta large in pixels", default 0). Method names get
_p1b<beta>[_pl<lambda>]. friends_layer2.py is unchanged; delete/archive this file if P1 is dropped.

(original header follows)
friends_layer2.py

Layer 2 defense against BrainWash: FRIENDS (Liu, Yang & Mirzasoleiman, NeurIPS 2022),
applied while the victim continual learner trains on the poisoned last task.

This is a deliberate fork of main_baselines.py (stage 4), NOT an edit of it, in the
same style as the Layer-1 scripts: the original reproduction stays byte-for-byte
reproducible, and every line that is not part of the original main_baselines.py is
marked "# >>> LAYER2" / "# <<< LAYER2". Without --friends this script does exactly
what main_baselines.py does (only the output location differs).

What --friends does (paper's Algorithm 2, on the poisoned task only):
  * warm-up epochs : train on the poisoned images, with fresh random noise r each step
  * then           : compute the friendly noise eta ONCE from the current model
                     (head of the task, KL divergence), for every training image
  * remaining      : train on image + eta, with fresh random noise r each step
The training code that does this is in friends_training.py (copies of each method's
train()/train_epoch() plus marked lines); the attacker's code and approaches/*.py are
not touched.

Extra options (work with or without --friends; they use the Layer-2 learner but add no noise by themselves):
    --bn_mode eval|nostats       BatchNorm control while training the task (see friends.py: FriendsDefense)
    --head_warmup_epochs N       the first N epochs train only the new head, body and BatchNorm frozen
    --log_old_acc                print old-task / current-task accuracy after every epoch

What the friendly noise preserves (and how it is optimized):
    --friendly_target head|features   head = the new task's logits, KL (the paper; options b and c);
                                 features = the shared 1024-d embedding, relative MSE (option a)
    --friendly_lambda / --friendly_lr / --friendly_momentum / --friendly_epochs
                                 the eta optimizer (official: 1 / 100 / 0.9 / 30); non-default values are
                                 tagged in the method name (_lam, _lr, _mom, _st)

Layer 1 + Layer 2 (DiffPure, then FRIENDS on the purified images):
    --purify --sigma_star 0.5    purify the task's training images (poisoned or clean, whichever arm this is)
                                 with the Layer-1 CIFAR purifier, diffpure_edm.py (repo root; imported, not
                                 modified), called like purify_cifar_layer1a.py does (chunk 256, num_steps 15,
                                 no extra seeding). The result is cached in --purify_cache_dir, keyed by the
                                 input pixels and the purification settings, so the first arm purifies and
                                 every later arm on the same images reuses it (and needs no diffusion model).
                                 --purify_ckpt is only needed when the cache does not have the images yet.

Arms, selected by flags (same flags as main_baselines.py for the data):
    clean                        --checkpoint <noise.pkl>
    poison                       --checkpoint <noise.pkl> --addnoise
    poison + FRIENDS             --checkpoint <noise.pkl> --addnoise --friends
    clean  + FRIENDS (cost)      --checkpoint <noise.pkl> --friends
    friendly noise only          ... --friends --components friendly_only --zeta 32
    random noise only            ... --friends --components random_only  --mu_rand 32
(the paper matches the budgets in the component ablation: 32 when a component runs alone)

Notation: eta = friendly noise, zeta = its per-pixel cap, r = random noise, mu = its
size (+-mu per pixel for Bernoulli), lambda = weight of the "make eta large" term.
epsilon is the BrainWash poison budget (0.1 / 0.3), as everywhere else in the thesis.

Results: acc_mat_*.npy and summary_layer2.tsv in --out_dir
(default: $LAYER2_OUT_DIR, else layer2_defense/results/runs).
"""
import sys, os, time
import numpy as np
import pickle as pkl

# >>> LAYER2: this file lives in layer2_defense/; the repo root (utils, resnet, approaches) and
# this folder (friends, friends_training) must be importable.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
# <<< LAYER2
import utils
import torch
from approaches.arguments import get_args
from resnet import ResNet18


tstart = time.time()


# >>> LAYER2: Layer 1 (DiffPure) in front of Layer 2. Uses the Layer-1 CIFAR purifier, diffpure_edm.py
# (load_edm_net / purify_in_chunks), as is; the purified images are cached so they are made only once.
def purify_task_inputs(x, args):
    import hashlib, inspect
    h = hashlib.sha1()
    h.update(x.detach().cpu().contiguous().numpy().tobytes())
    h.update(repr((float(args.sigma_star), args.purify_seed, int(args.purify_steps), int(args.purify_chunk))).encode())
    key = h.hexdigest()[:16]
    cache_dir = args.purify_cache_dir or os.path.join(args.out_dir, 'purify_cache')
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'purified_edm_{key}.pt')
    t0 = time.time()
    if os.path.isfile(path):
        print(f'[LAYER1] reusing cached purified images: {path}')
        xp = torch.load(path, map_location='cpu')
    else:
        if args.purify_ckpt is None:
            sys.exit(f'[LAYER1] no cached purified images for these inputs ({path}) and no --purify_ckpt to make them')
        from diffpure_edm import load_edm_net, purify_in_chunks
        kw = {'device': 'cuda'} if 'device' in inspect.signature(load_edm_net).parameters else {}
        net = load_edm_net(args.purify_ckpt, **kw)
        params = inspect.signature(purify_in_chunks).parameters
        wanted = dict(chunk_size=args.purify_chunk, num_steps=args.purify_steps, device='cuda')
        if args.purify_seed is not None:      # default None = like the Layer-1 CIFAR script: no seed, the global RNG state is used
            wanted['seed'] = args.purify_seed
        kw = {k: v for k, v in wanted.items() if k in params}
        if args.purify_seed is not None and 'seed' not in params:
            torch.manual_seed(args.purify_seed)
        print(f'[LAYER1] purifying {tuple(x.shape)} images: sigma_star={args.sigma_star}, call args {kw}, '
              f'purify_seed={args.purify_seed}')
        xp = purify_in_chunks(net, x.detach().cpu(), args.sigma_star, **kw)
        del net
        torch.cuda.empty_cache()
        torch.save(xp, path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f'[LAYER1] cached purified images -> {path}')
    if tuple(xp.shape) != tuple(x.shape):
        sys.exit(f'[LAYER1] purified images have shape {tuple(xp.shape)}, expected {tuple(x.shape)}')
    d = xp.float() - x.detach().cpu().float()
    print(f'[LAYER1] purification changed the inputs by Linf={d.abs().max().item():.4f}, '
          f'mean L2={d.flatten(1).norm(dim=1).mean().item():.4f}  [{time.time() - t0:.1f}s]')
    return xp.to(device=x.device, dtype=x.dtype)
# <<< LAYER2

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

    # >>> LAYER2: with --friends (or any of the other Layer-2 options), the Layer-2 version of the learner
    # (friends_training.py)
    layer2 = args.friends or args.bn_mode != 'normal' or args.head_warmup_epochs > 0 or args.log_old_acc
    if layer2:
        from friends_training import TRAINERS
        Appr = TRAINERS[args.approach]
    else:
        Appr = approach.Appr
    # <<< LAYER2

    if 'afec' in args.approach:
        if args.checkpoint is not None:
            lamb = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb']
            lamb_emp = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb_emp']
        else:
            lamb = args.lamb
            lamb_emp = args.lamb_emp

        appr = Appr(net, sbatch=args.batch_size, lamb=lamb, lamb_emp=lamb_emp,
                            lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name,
                            empty_net = net_emp, clipgrad=args.clip)
    else:
        if args.checkpoint is not None:
            lamb = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb']
        else:
            lamb = args.lamb

        appr = Appr(net, lamb=lamb, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, clipgrad=args.clip)


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



    # >>> LAYER2: record the hardware and software (the same code gave different early-training results in different
    # Colab sessions); then the FRIENDS settings; attach() hands the random noise to the learner
    print(f'[LAYER2] torch {torch.__version__}, cuda {torch.version.cuda}, device {torch.cuda.get_device_name(0)}')
    defense = None
    if layer2:
        from friends_p1 import FriendsDefenseP1
        defense = FriendsDefenseP1(p1_beta=args.p1_beta, p1_lambda=args.p1_lambda,
                                 zeta255=args.zeta, mu255=args.mu_rand, kind=args.noise_kind,
                                 warmup_epochs=args.warmup_epochs, friendly_epochs=args.friendly_epochs,
                                 friendly_lr=args.friendly_lr, friendly_lambda=args.friendly_lambda,
                                 friendly_batch=args.friendly_batch,
                                 use_friendly=args.friends and args.components in ('both', 'friendly_only'),
                                 use_random=args.friends and args.components in ('both', 'random_only'),
                                 random_start_epoch=args.warmup_epochs if args.random_after_warmup else 0,
                                 bn_mode=args.bn_mode, head_warmup_epochs=args.head_warmup_epochs,
                                 log_old_acc=args.log_old_acc, seed=args.seed, device='cuda',
                                 friendly_target=args.friendly_target, friendly_loss=args.friendly_loss,
                                 friendly_momentum=args.friendly_momentum)
        defense.attach(appr)
        if args.friends:
            print(f'[LAYER2] FRIENDS on: components={args.components} zeta={args.zeta}/255 mu={args.mu_rand}/255 '
                  f'({args.noise_kind}) warm-up={args.warmup_epochs} epoch(s) of {args.nepochs}; '
                  f'random noise from epoch {args.warmup_epochs if args.random_after_warmup else 0}; '
                  f'friendly noise: {args.friendly_epochs} steps, lr={args.friendly_lr}, momentum={args.friendly_momentum}, '
                  f'lambda={args.friendly_lambda}, '
                  f'preserves {args.friendly_target} (loss {args.friendly_loss or ("KL" if args.friendly_target == "head" else "MSE_rel")})')
        print(f'[LAYER2] BatchNorm: {args.bn_mode}; head-only epochs: {args.head_warmup_epochs}; '
              f'per-epoch accuracy log: {args.log_old_acc}')
    # <<< LAYER2

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

                xtrain = torch.clamp(xtrain + all_noise, 0, 1)


            else:
                print('Using noise from checkpoint')
                all_noise = checkpoint_dict['latest_noise']
                noise_prm = checkpoint_dict['rnd_idx_train']
                xtrain = xtrain[noise_prm]
                ytrain = ytrain[noise_prm]


                xtrain = torch.clamp(xtrain + all_noise, 0, 1)

        # >>> LAYER2: Layer 1 (DiffPure) on this task's training images, before any Layer-2 training
        if args.purify:
            xtrain = purify_task_inputs(xtrain, args)
        # <<< LAYER2

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

        # >>> LAYER2: keep FRIENDS outputs apart from the baseline's acc_mat_*.npy
        if args.purify:
            method += f'_pur{args.sigma_star:g}'
        if args.friends:
            method = f'{method}_friends-{args.components}_z{args.zeta:g}_mu{args.mu_rand:g}_w{args.warmup_epochs}'
            if args.components != 'random_only':
                method += f'_p1b{args.p1_beta:g}' + (f'_pl{args.p1_lambda:g}' if args.p1_lambda else '')
            if args.friendly_target != 'head' and args.components != 'random_only':
                method += f'_tgt-{args.friendly_target}'
            if args.components != 'random_only':   # eta optimizer settings, tagged only when not the official defaults
                for flag, val, dflt in (('lam', args.friendly_lambda, 1.), ('lr', args.friendly_lr, 100.),
                                        ('mom', args.friendly_momentum, 0.9), ('st', args.friendly_epochs, 30)):
                    if float(val) != float(dflt):
                        method += f'_{flag}{val:g}'
            if args.random_after_warmup:
                method += '_ra'
        if args.bn_mode != 'normal':
            method += f'_bn-{args.bn_mode}'
        if args.head_warmup_epochs > 0:
            method += f'_hw{args.head_warmup_epochs}'
        os.makedirs(args.out_dir, exist_ok=True)
        # the stage-3 noise file names are ~170 characters; file systems (and Google Drive) allow 255 per
        # name, so a long name is cut and tagged with a short hash (the full name stays in summary_layer2.tsv)
        import hashlib
        short_name = acc_mat_sace_name
        if len(short_name) > 60:
            short_name = short_name[:50] + '_' + hashlib.md5(short_name.encode()).hexdigest()[:8]
        out_npy = os.path.join(args.out_dir, f'acc_mat_{short_name}_{method}_s{args.seed}.npy')
        np.save(out_npy, acc)
        print(f'[LAYER2] saved accuracy matrix -> {out_npy}')
        # <<< LAYER2



    bwt_before = np.mean((acc[args.lasttask-1] - np.diag(acc))[:args.lasttask-1][:-1])
    avg_acc_before  = np.mean(acc[args.lasttask-1, :args.lasttask])

    if args.checkpoint is not None:
        bwt_after = np.mean((acc[-1] - np.diag(acc))[:-1])
        avg_acc_after  = np.mean(acc[-1][:-1])
        last_task_acc = acc[-1, -1]
        print(f'After BWT : {bwt_after} After avg acc : {avg_acc_after} Last task acc : {last_task_acc}')

        # >>> LAYER2: one-line machine-readable summary next to the .npy
        summary = os.path.join(args.out_dir, 'summary_layer2.tsv')
        new = not os.path.isfile(summary)
        L = defense.log if defense is not None else {}
        fmt = lambda k, f: (f % L[k]) if k in L else '-'
        with open(summary, 'a') as fh:
            if new:
                fh.write('method\tapproach\texperiment\tseed\tfriends\tcomponents\tzeta255\tmu255\twarmup_epochs\t'
                         'bwt_after\tavg_acc_after\tlast_task_acc\told_acc_at_eta\teta_mean_over_zeta\teta_frac_at_bound\t'
                         'noise_pkl\n')
            fh.write(f'{method}\t{args.approach}\t{args.experiment}\t{args.seed}\t{int(args.friends)}\t'
                     f'{args.components if args.friends else "-"}\t{args.zeta if args.friends else "-"}\t'
                     f'{args.mu_rand if args.friends else "-"}\t{args.warmup_epochs if args.friends else "-"}\t'
                     f'{bwt_after:.4f}\t{avg_acc_after:.4f}\t{last_task_acc:.4f}\t'
                     f'{fmt("old_task_acc_at_eta", "%.2f")}\t{fmt("eta_mean_over_zeta", "%.4f")}\t'
                     f'{fmt("eta_frac_at_bound", "%.4f")}\t{os.path.basename(args.checkpoint)}\n')
        print(f'[LAYER2] appended summary row -> {summary}')
        # <<< LAYER2


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
    # >>> LAYER2: parse the defense flags ourselves so approaches/arguments.py never has to be
    # touched; they are removed from sys.argv before the original get_args() sees it.
    import argparse
    _p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _p.add_argument('--friends', action='store_true', help='turn the FRIENDS defense on')
    _p.add_argument('--components', default='both', choices=['both', 'friendly_only', 'random_only'],
                    help='both = FRIENDS; friendly_only / random_only = component ablation')
    _p.add_argument('--zeta', type=float, default=16, help='cap of the friendly noise per pixel, x/255 (paper: 16)')
    _p.add_argument('--mu_rand', type=float, default=16, help='size of the random noise, x/255 (paper: 16)')
    _p.add_argument('--noise_kind', default='bernoulli', choices=['bernoulli', 'uniform', 'gaussian'])
    _p.add_argument('--warmup_epochs', type=int, default=2,
                    help='epochs of random-noise-only training before the friendly noise is computed '
                         '(paper: 5 of 80)')
    _p.add_argument('--random_after_warmup', action='store_true',
                    help='keep the warm-up plain: random noise switches on together with the friendly noise '
                         '(the paper has it on from the first epoch)')
    _p.add_argument('--bn_mode', default='normal', choices=['normal', 'eval', 'nostats'],
                    help="BatchNorm control while training the task: 'eval' = use and keep the stored (checkpoint) "
                         "statistics; 'nostats' = normalize with batch statistics but never update the stored ones")
    _p.add_argument('--head_warmup_epochs', type=int, default=0,
                    help='first N epochs train only the new head (body and BatchNorm frozen); with --friends '
                         'use --warmup_epochs >= N so the friendly noise is computed after them')
    _p.add_argument('--log_old_acc', action='store_true',
                    help='print old-task and current-task test accuracy after every epoch')
    _p.add_argument('--friendly_target', default='head', choices=['head', 'features'],
                    help="what the friendly noise must leave unchanged: 'head' = the new task's logits (the paper; options "
                         "b and c); 'features' = the shared 1024-d embedding the old heads read (option a)")
    _p.add_argument('--friendly_loss', default=None, choices=['KL', 'MSE', 'MSE_rel'],
                    help='default: KL for --friendly_target head, MSE_rel (scale-free MSE) for features')
    _p.add_argument('--friendly_epochs', type=int, default=30, help='optimization steps for the friendly noise')
    _p.add_argument('--friendly_lr', type=float, default=100., help='learning rate of the eta optimizer (official: 100)')
    _p.add_argument('--friendly_momentum', type=float, default=0.9, help='SGD momentum of the eta optimizer (official: 0.9)')
    _p.add_argument('--friendly_lambda', type=float, default=1., help='weight of the "make eta large" term')
    _p.add_argument('--friendly_batch', type=int, default=128)
    _p.add_argument('--out_dir', default=os.environ.get('LAYER2_OUT_DIR', os.path.join(HERE, 'results', 'runs')),
                    help='where acc_mat_*.npy and summary_layer2.tsv are written')
    _p.add_argument('--purify', action='store_true',
                    help='Layer 1 first: purify the task training images with diffpure_edm.py (DiffPure)')
    _p.add_argument('--sigma_star', type=float, default=0.5, help='DiffPure strength (Layer 1 used 0.5 for CIFAR/EWC)')
    _p.add_argument('--purify_ckpt', default=None, help='EDM checkpoint for diffpure_edm.load_edm_net '
                    '(only needed when the cache does not have the purified images yet)')
    _p.add_argument('--purify_cache_dir', default=None, help='where purified images are cached (default: <out_dir>/purify_cache)')
    _p.add_argument('--purify_seed', type=int, default=None, help='seed of the purification noise; default None = no extra '
                    'seeding, exactly like purify_cifar_layer1a.py (the noise then follows from --seed)')
    _p.add_argument('--purify_chunk', type=int, default=256, help='images per purification batch (Layer 1 CIFAR: 256)')
    _p.add_argument('--purify_steps', type=int, default=15, help='purification steps (Layer 1 CIFAR: num_steps=15)')
    _p.add_argument('--p1_beta', type=float, default=1.0, help='P1: weight of "disturb the shared features"')
    _p.add_argument('--p1_lambda', type=float, default=0.0, help='P1: weight of "make eta large in pixels" (FRIENDS term)')
    _dargs, _rest = _p.parse_known_args()
    sys.argv = [sys.argv[0]] + _rest
    # <<< LAYER2

    args = get_args()

    # >>> LAYER2
    for _k, _v in vars(_dargs).items():
        setattr(args, _k, _v)
    if args.friends and not 0 <= args.warmup_epochs < args.nepochs:
        sys.exit(f'--warmup_epochs must be in [0, {args.nepochs - 1}] (nepochs={args.nepochs})')
    _l2 = args.friends or args.bn_mode != 'normal' or args.head_warmup_epochs > 0 or args.log_old_acc
    if _l2 and args.approach not in ('ewc', 'rwalk', 'afec_ewc'):
        sys.exit(f'the Layer-2 options support ewc, rwalk and afec_ewc (got {args.approach})')
    if args.friendly_target == 'features' and args.friendly_loss == 'KL':
        sys.exit('--friendly_loss KL needs logits: use MSE_rel (default) or MSE with --friendly_target features')
    if not 0 <= args.head_warmup_epochs < args.nepochs:
        sys.exit(f'--head_warmup_epochs must be in [0, {args.nepochs - 1}] (nepochs={args.nepochs})')
    if args.friends and args.components != 'random_only' and 0 < args.head_warmup_epochs > args.warmup_epochs:
        sys.exit('--head_warmup_epochs must be <= --warmup_epochs (the friendly noise is computed after the head-only epochs)')
    # <<< LAYER2

    main(args)
