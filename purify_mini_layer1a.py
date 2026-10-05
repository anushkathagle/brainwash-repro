"""
purify_mini_layer1a.py

Layer 1a defense against BrainWash, for **miniImageNet (84x84)**: DiffPure-style
purification of the poisoned task's training images, applied before continual
learning proceeds.

This is the miniImageNet counterpart of purify_cifar_layer1a.py. Same design
rule: it is a deliberate fork of main_baselines.py, NOT an edit of it, so the
original reproduction stays byte-for-byte reproducible. Every line that is not
part of the original main_baselines.py is marked "# >>> LAYER1A" / "# <<< LAYER1A".

Differences from the CIFAR version, and why:
  * Purifier backbone is OpenAI guided-diffusion ImageNet-256 UNCONDITIONAL
    (256x256_diffusion_uncond.pt) instead of a fine-tuned EDM net. No diffusion
    model exists for miniImageNet, and the only released unconditional ImageNet
    models are ImageNet-64 and ImageNet-256. Images are bicubic-resized
    84 -> 256 for purification and back to 84 afterwards.
  * Strength is --sigma_star, SAME NOTATION as purify_cifar_layer1a.py (an EDM-
    style noise stddev). Internally converted to the nearest matching DDPM
    timestep via diffpure_guided.py:sigma_to_t (guided-diffusion is a VP model,
    parameterized by discrete timesteps, not sigma directly -- see that module's
    docstring for the conversion and why a bare rename would be wrong).
  * --sigma_star 0 with --purify gives the RESIZE-ONLY control arm (see below).
  * acc_mat_*.npy is written to --out_dir instead of the repo root.

Arms this script can run, selected purely by CLI flags:
    Arm A (clean, no purify):    --checkpoint <noise.pkl>
    Arm B (poison, no purify):   --checkpoint <noise.pkl> --addnoise
    Arm B'(uniform, no purify):  --checkpoint <noise.pkl> --addnoise --uniform
    Arm C (poison + purify):     --checkpoint <noise.pkl> --addnoise --purify --sigma_star 0.5 --purify_ckpt <...pt>
    Arm D (clean + purify):      --checkpoint <noise.pkl> --purify --sigma_star 0.5 --purify_ckpt <...pt>
    Arm E (poison + RESIZE ONLY):--checkpoint <noise.pkl> --addnoise --purify --sigma_star 0 --purify_ckpt <...pt>

Arm E is not optional book-keeping. Bicubic 84->256->84 is itself a low-pass
filter that strips part of an L-inf perturbation, so without Arm E you cannot
claim a robustness gain comes from diffusion rather than from resampling.
Arm D is how you pick sigma_star: choose the largest sigma_star whose clean-
accuracy cost stays inside your budget, THEN report robustness at that fixed
sigma_star. Tuning sigma_star on the attacked arm is tuning on the test
condition. Pick it independently for mini and CIFAR -- see the module docstring
of diffpure_guided.py for why the same sigma_star number is not guaranteed to
be equally strong on both backbones.

Requirements beyond the base brainwash-repro environment:
    - diffpure_guided.py in this same directory (repo root).
    - `guided_diffusion` importable (it is CODE, not part of the checkpoint):
          git clone https://github.com/openai/guided-diffusion.git
          export PYTHONPATH=$PYTHONPATH:/path/to/guided-diffusion
    - the checkpoint 256x256_diffusion_uncond.pt downloaded locally.
"""

import sys, os, time, hashlib
import numpy as np
import pickle as pkl
import utils
import torch
from approaches.arguments import get_args
from resnet import ResNet18

# >>> LAYER1A: purifier import
from diffpure_guided import load_guided_net, purify_in_chunks, cache_key
# <<< LAYER1A

tstart = time.time()

# >>> LAYER1A: default output location for acc_mat_*.npy (kept off the repo root
# and off the 16 GB home quota).
DEFAULT_OUT_DIR = ('/storage/work/axt5884/projects/myworkflow/brainwash-repro/'
                   'repro/artifacts/miniImagenet_files/purify_results')
# <<< LAYER1A


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

    # >>> LAYER1A: approaches/data_utils.py's generate_split_mini_imagenet_tasks
    # (and possibly generate_split_tiny_imagenet_tasks) hardcode data[n]['name']
    # = 'cifar100' and inputsize = [3,32,32] -- copy-paste leftovers from the
    # CIFAR loader. Both are dead metadata: `name` only feeds print statements
    # ("Skip task N : cifar100", "Task 9 (cifar100)") and `inputsize` is passed
    # to appr.train() but never used for shape logic (the ResNet uses
    # adaptive_avg_pool2d, so it adapts to whatever the real tensor shape is --
    # confirmed elsewhere in these logs by diffpure_guided.py printing the
    # ACTUAL tensor shape, e.g. "purifying 5000 images ... at 84x84", which is
    # correct even when this label says cifar100/32x32).
    #
    # Relabeled HERE rather than editing the shared approaches/data_utils.py,
    # so the original reproduction pipeline (main_baselines.py, and anything
    # else importing that module) stays byte-for-byte untouched -- this fork
    # only changes what ITS OWN logs print, nothing computational.
    if args.experiment != 'split_cifar100':
        _dataset_label = {'split_mini_imagenet': 'miniimagenet',
                          'split_tiny_imagenet': 'tinyimagenet'}[args.experiment]
        for _t in data:
            if isinstance(_t, int):
                data[_t]['name'] = _dataset_label
        inputsize = [3, im_sz, im_sz]
    # <<< LAYER1A

    print('\nInput size =', inputsize, '\nTask info =', taskcla)


    ########################################################################################################################

    print('Inits...')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    nf = 32

    net = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()
    net_emp = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()

    # >>> LAYER1A: load the purifier once, outside the per-task loop.
    # Note --sigma_star 0 is still a real arm (resize-only control), so the gate is
    # --purify, not sigma_star > 0.
    purify_net = None
    if args.purify:
        assert args.purify_ckpt is not None, '--purify_ckpt required when --purify is set'
        print(f'[LAYER1A] Loading purifier from {args.purify_ckpt} (sigma_star={args.sigma_star}) ...')
        purify_net = load_guided_net(args.purify_ckpt, device='cuda',
                                     use_fp16=not args.purify_fp32)
    # <<< LAYER1A

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

        # >>> LAYER1A: purify xtrain (poisoned, uniform or clean, whichever arm this is).
        # This sits AFTER the rnd_idx_train permutation and the noise addition, so it
        # is a pure input transform and needs to know nothing about either.
        if args.purify:
            t0 = time.time()
            cached = None
            resume_path = None
            if args.purify_cache_dir:
                os.makedirs(args.purify_cache_dir, exist_ok=True)
                key = cache_key(xtrain, args.sigma_star, args.purify_resize, args.purify_seed)
                cached = os.path.join(args.purify_cache_dir, f'purified_{key}.pt')
                # >>> LAYER1A: RESUMABILITY. Progress is checkpointed to resume_path every
                # `--purify_ckpt_every` chunks inside purify_in_chunks. If this job dies
                # mid-purification (wall time limit, a dropped salloc session -- anything),
                # re-running the EXACT SAME command picks up from the last checkpointed
                # chunk instead of starting over. Requires --purify_cache_dir to be set;
                # without it there's nowhere durable to write progress to.
                resume_path = os.path.join(args.purify_cache_dir, f'resume_{key}.pt')
                # <<< LAYER1A

            if cached is not None and os.path.isfile(cached):
                print(f'[LAYER1A] Reusing cached purified tensor: {cached}')
                xtrain_p = torch.load(cached, map_location='cpu')
            else:
                print(f'[LAYER1A] Purifying task {t} inputs: sigma_star={args.sigma_star}, '
                      f'resize={args.purify_resize}, chunk={args.purify_chunk}')
                if resume_path and os.path.isfile(resume_path):
                    print(f'[LAYER1A] Found a resume checkpoint -> {resume_path} '
                          f'(picking up where the last run left off)')
                xtrain_p = purify_in_chunks(purify_net, xtrain.cpu(), args.sigma_star,
                                            chunk_size=args.purify_chunk,
                                            device='cuda', seed=args.purify_seed,
                                            resume_path=resume_path,
                                            checkpoint_every=args.purify_ckpt_every)
                if cached is not None:
                    torch.save(xtrain_p, cached)
                    print(f'[LAYER1A] Cached purified tensor -> {cached}')

            # Report how much the defense actually moved the pixels. A near-zero
            # L-inf here means purification silently did nothing.
            delta_linf = (xtrain_p - xtrain.cpu()).abs().max().item()
            delta_l2 = (xtrain_p - xtrain.cpu()).flatten(1).norm(dim=1).mean().item()
            print(f'[LAYER1A] purification changed inputs by Linf={delta_linf:.4f}, '
                  f'mean L2={delta_l2:.4f}  [{time.time() - t0:.1f}s]')

            xtrain = xtrain_p.to(xtrain.device)
        # <<< LAYER1A

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
        # >>> LAYER1A: use the noise DIRECTORY name (the NOISE_TAG, e.g.
        # "mini_afec_ewc_cautious_eps0.1") instead of the raw checkpoint
        # filename here. AFEC's pkl names encode wcur/mode/every hyperparam
        # TWICE and run 150+ chars on their own; concatenating the purify
        # suffix on top pushes the final path component past Linux's 255-byte
        # NAME_MAX and np.save() crashes with ENAMETOOLONG -- AFTER an entire
        # purify+train+eval run has already completed, losing the result.
        # The directory name is already the canonical short identifier used
        # everywhere else (EVAL_TAG, stage4_purify.sbatch's NOISE_TAG lookup),
        # and since stage3.sbatch keeps only the newest pkl per directory,
        # directory name <-> single pkl is already a 1:1 mapping elsewhere in
        # this pipeline, so reusing it here introduces no new assumption.
        # A hash-truncate fallback guards against any name still being too
        # long for any reason (e.g. a very long --out_dir path on some other
        # filesystem), so this can never crash on ENAMETOOLONG again.
        acc_mat_sace_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        if not acc_mat_sace_name or acc_mat_sace_name in ('.', '/'):
            acc_mat_sace_name = os.path.basename(args.checkpoint)[:-4]  # .pkl stripped
        if len(acc_mat_sace_name) > 100:
            h = hashlib.sha1(acc_mat_sace_name.encode()).hexdigest()[:10]
            acc_mat_sace_name = acc_mat_sace_name[:80] + '_' + h
        # <<< LAYER1A

        if args.addnoise is False:
            method = 'clean'
        if args.addnoise and args.uniform:
            method = 'uniform'
        elif args.addnoise and args.uniform is False:
            method = 'ours'

        # >>> LAYER1A: keep purified-arm outputs from overwriting the baseline's
        # acc_mat_*.npy, and write everything to --out_dir instead of the repo root.
        if args.purify:
            tag = 'resizeonly' if float(args.sigma_star) == 0.0 else f'sigma{args.sigma_star}'
            method = f'{method}_purify_{tag}_r{args.purify_resize}'

        os.makedirs(args.out_dir, exist_ok=True)
        out_npy = os.path.join(args.out_dir, f'acc_mat_{acc_mat_sace_name}_{method}.npy')
        # Defensive: a failure here (ENAMETOOLONG or anything else) must NEVER
        # take the After-BWT computation / summary.tsv append down with it --
        # those are the numbers that actually matter, and losing them after a
        # multi-hour run to a save-path error is the exact failure this guards
        # against. Worst case on failure: the .npy is missing but everything
        # else (console log, summary.tsv row) still gets written.
        try:
            np.save(out_npy, acc)
            print(f'[LAYER1A] saved accuracy matrix -> {out_npy}')
        except OSError as e:
            print(f'[LAYER1A] WARNING: failed to save accuracy matrix to {out_npy} ({e}) '
                  f'-- continuing anyway so After BWT / summary.tsv are not lost. '
                  f'The full per-task accuracy matrix is printed above (search "Accuracies =").')
        # <<< LAYER1A



    bwt_before = np.mean((acc[args.lasttask-1] - np.diag(acc))[:args.lasttask-1][:-1])
    avg_acc_before  = np.mean(acc[args.lasttask-1, :args.lasttask])

    if args.checkpoint is not None:
        bwt_after = np.mean((acc[-1] - np.diag(acc))[:-1])
        avg_acc_after  = np.mean(acc[-1][:-1])
        last_task_acc = acc[-1, -1]
        print(f'After BWT : {bwt_after} After avg acc : {avg_acc_after} Last task acc : {last_task_acc}')

        # >>> LAYER1A: one-line machine-readable summary next to the .npy, so the
        # arms can be collected later without re-parsing the full SLURM log.
        summary = os.path.join(args.out_dir, 'summary.tsv')
        new = not os.path.isfile(summary)
        with open(summary, 'a') as fh:
            if new:
                fh.write('method\tapproach\texperiment\tseed\tsigma_star\tresize\t'
                         'bwt_after\tavg_acc_after\tlast_task_acc\tnoise_pkl\n')
            fh.write(f'{method}\t{args.approach}\t{args.experiment}\t{args.seed}\t'
                     f'{args.sigma_star if args.purify else "-"}\t'
                     f'{args.purify_resize if args.purify else "-"}\t'
                     f'{bwt_after:.4f}\t{avg_acc_after:.4f}\t{last_task_acc:.4f}\t'
                     f'{os.path.basename(args.checkpoint)}\n')
        print(f'[LAYER1A] appended summary row -> {summary}')
        # <<< LAYER1A


    print(f'Before BWT : {bwt_before} Before avg acc : {avg_acc_before}')

    save_dict['last_task'] = int(args.lasttask)
    save_dict['acc_mat'] = acc
    save_dict['avg_acc'] = np.mean(acc[-1, :args.lasttask])
    save_dict['bwt'] = bwt_before
    save_dict['model'] = net.state_dict()
    if 'afec' in args.approach:
        save_dict['cont_method_args']['model_emp'] = net_emp.state_dict()

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
    # >>> LAYER1A: parse the defense flags ourselves so approaches/arguments.py
    # never has to be touched. We strip them out of sys.argv before handing the
    # rest to the original get_args().
    import argparse
    _defense_parser = argparse.ArgumentParser(add_help=False)
    _defense_parser.add_argument('--purify', action='store_true',
                                 help='enable DiffPure purification of the poisoned task inputs')
    _defense_parser.add_argument('--sigma_star', default=0.5, type=float,
                                 help='EDM-style noise stddev, same notation as purify_cifar_layer1a.py. '
                                      'Converted internally to the nearest matching DDPM timestep under '
                                      "this model's VP schedule (see diffpure_guided.py:sigma_to_t) -- "
                                      'the resolved timestep/sigma is printed at run time. Default 0.5 '
                                      '~ t~144/1000 under the linear/1000-step schedule. '
                                      '0 = resize-only control arm.')
    _defense_parser.add_argument('--purify_ckpt', default=None, type=str,
                                 help='path to 256x256_diffusion_uncond.pt')
    _defense_parser.add_argument('--purify_resize', default=256, type=int,
                                 help="resolution the purifier runs at (256 for the guided-diffusion model)")
    _defense_parser.add_argument('--purify_chunk', default=32, type=int,
                                 help='images per purification batch (lower if OOM)')
    _defense_parser.add_argument('--purify_seed', default=None, type=int,
                                 help='seed for the stochastic purification (default: --seed)')
    _defense_parser.add_argument('--purify_fp32', action='store_true',
                                 help='run the purifier in fp32 instead of fp16 (slower, more memory)')
    _defense_parser.add_argument('--purify_cache_dir', default=None, type=str,
                                 help='dir to cache purified tensors, keyed by input pixels + sigma_star. '
                                      'ALSO enables resumability (see --purify_ckpt_every): without this set, '
                                      'a killed run has nowhere to save progress and restarts from image 0.')
    _defense_parser.add_argument('--purify_ckpt_every', default=1, type=int,
                                 help='checkpoint progress every N chunks when --purify_cache_dir is set '
                                      '(default: every chunk). Only matters for how much work is lost if '
                                      'the job dies mid-purification -- lower is safer, costs a bit more I/O.')
    _defense_parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, type=str,
                                 help='where to write acc_mat_*.npy and summary.tsv')
    _defense_args, _remaining_argv = _defense_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining_argv
    # <<< LAYER1A

    args = get_args()

    # >>> LAYER1A
    for _k, _v in vars(_defense_args).items():
        setattr(args, _k, _v)
    if args.purify_seed is None:
        args.purify_seed = args.seed
    # <<< LAYER1A

    main(args)
