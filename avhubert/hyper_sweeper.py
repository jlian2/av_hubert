import copy
import itertools
import os
import os.path as op
import re
import subprocess
import sys
from omegaconf import OmegaConf


class HyperParam(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def set(self, key, val):
        setattr(self, key, val)

    def flatten(self):
        keys_flat, vals_flat = [[]], [[]]
        for key, vals in self.__dict__.items():
            N = len(keys_flat)
            for i in range(len(vals) - 1):
                for n in range(N):
                    keys_flat.append(copy.deepcopy(keys_flat[n]))
                    vals_flat.append(copy.deepcopy(vals_flat[n]))
            for index in range(N * len(vals)):
                iy = int(index / N)
                keys_flat[index].append(key)
                vals_flat[index].append(vals[iy])
        return keys_flat, vals_flat


def get_scripts(
    run_dir,
    script_dir,
    test_modality,
    subsets,
    max_tokens=800,
    name_level=1,
    is_best=True,
    noise_dir=None,
    noise_subdir=None,
    snr=None,
    beams=[10],
    lenpens=[0],
):
    ckpt = "best" if is_best else "last"
    print(
        f"Using {ckpt} checkpoint for {run_dir} on {test_modality} with {noise_subdir} noise at {snr} dB"
    )
    env_header = (
        "#! /bin/bash\n"
        # "source ~/.bashrc\n"
        # "conda activate subm-fairseq\n"
    )
    run_dir = re.sub("//*$", "", run_dir)
    run_dir = re.sub("//*", "/", run_dir)
    model_dir = op.join(run_dir, f"checkpoints/checkpoint_{ckpt}.pt")
    # config_fn = op.join(run_dir, ".hydra/config.yaml")
    # config = OmegaConf.load(config_fn)
    # max_tokens = 800 if config.task.modalities[0] == "video" else 5000
    print(f"Max tokens: {max_tokens}")

    if test_modality == "av":
        mod_str = "['video','audio']"
    elif test_modality == "a":
        mod_str = "['audio']"
    elif test_modality == "v":
        mod_str = "['video']"
    else:
        raise ValueError("modality {test_modality} not supported")

    run_scripts = []
    for gen_subset in subsets:
        cmd_prefixes, job_prefixes = [], []

        if (noise_dir is None) or (noise_subdir is None) or (snr is None):
            result_dir = op.join(run_dir, f"results_{test_modality}_{gen_subset}")
            cmd_prefix = (
                f"python -B infer_s2s.py --config-dir ./conf/ "
                f"--config-name s2s_decode dataset.gen_subset={gen_subset} "
                f"dataset.max_tokens={max_tokens} "
                f"common_eval.path={run_dir}/checkpoints/checkpoint_{ckpt}.pt "
                f"common_eval.results_path={result_dir}/ "
                f"common.user_dir=`pwd` override.modalities={mod_str} "
            )
            job_prefix = (
                f"{'_'.join(run_dir.split('/')[-name_level:])}_"
                f"{test_modality}_{gen_subset}"
            )
        else:
            # for noise_subdir in ["babble", "speech", "music", "noise"]:
            #     for snr in [-10, -5, 0, 5, 10]:
            result_dir = op.join(
                run_dir,
                f"results_{test_modality}_{gen_subset}_{noise_subdir}_{snr}",
            )
            cmd_prefix = (
                f"python -B infer_s2s.py --config-dir ./conf/ "
                f"--config-name s2s_decode dataset.gen_subset={gen_subset} "
                f"dataset.max_tokens={max_tokens} "
                f"common_eval.path={run_dir}/checkpoints/checkpoint_{ckpt}.pt "
                f"common_eval.results_path={result_dir}/ "
                f"common.user_dir=`pwd` override.modalities={mod_str} "
                f"override.noise_prob=1 override.noise_snr={snr} override.noise_wav={noise_dir}/{noise_subdir}/ "
            )
            job_prefix = (
                f"{'_'.join(run_dir.split('/')[-name_level:])}_"
                f"{test_modality}_{gen_subset}_{noise_subdir}_{snr}"
            )
        cmd_prefixes.append(cmd_prefix)
        job_prefixes.append(job_prefix)

        hyp = HyperParam()
        hyp.set("generation.beam", beams)
        hyp.set("generation.lenpen", lenpens)

        num_per_task = 5
        param_keys_list, param_vals_list = hyp.flatten()
        for cmd_prefix, job_prefix in zip(cmd_prefixes, job_prefixes):
            cmds = []
            for param_keys, param_vals in zip(param_keys_list, param_vals_list):
                assert len(param_keys) == len(param_vals)
                cmd = cmd_prefix
                for param_key, param_val in zip(param_keys, param_vals):
                    if isinstance(param_val, list):
                        param_val = f"[{','.join(str(x) for x in param_val)}]"
                    cmd = f"{cmd} {param_key}={param_val}"
                cmds.append(f"{cmd}\n")

            for i in range(0, len(cmds), num_per_task):
                job_id = str(i // num_per_task)
                run_script = op.join(script_dir, f"{job_prefix}_{job_id}.sh")
                with open(run_script, "w") as f:
                    f.write("\n".join([env_header] + cmds[i : i + num_per_task]))
                run_scripts.append(run_script)

    return run_scripts


def batch_get_scripts(
    run_dirs,
    script_dir,
    test_modalities,
    subsets,
    name_level,
    sbatch_args,
    noise_dir,
    noise_subdirs,
    snrs,
    beams,
    lenpens,
    max_tokens,
):
    is_best = True
    run_dirs = [op.abspath(x) for x in run_dirs]
    run_scripts = []
    os.makedirs(script_dir, exist_ok=True)
    for run_dir in run_dirs:
        for test_modality in test_modalities:
            run_scripts.extend(
                get_scripts(
                    run_dir=run_dir,
                    script_dir=script_dir,
                    test_modality=test_modality,
                    subsets=subsets,
                    max_tokens=max_tokens,
                    name_level=name_level,
                    is_best=is_best,
                )
            )

        for test_modality in set(test_modalities).intersection({"a", "av"}):
            for noise_subdir, snr in itertools.product(noise_subdirs, snrs):
                run_scripts.extend(
                    get_scripts(
                        run_dir=run_dir,
                        script_dir=script_dir,
                        test_modality=test_modality,
                        subsets=subsets,
                        max_tokens=max_tokens,
                        name_level=name_level,
                        is_best=is_best,
                        noise_dir=noise_dir,
                        noise_subdir=noise_subdir,
                        snr=snr,
                    )
                )

    launch_script_name = op.join(script_dir, "launch.sh")
    with open(launch_script_name, "w") as fo:
        for run_script in run_scripts:
            fo.write(f"sbatch {sbatch_args} {run_script}\n")

    subprocess.call(f"chmod +x {script_dir}/*sh", shell=True)
    print(f"run {launch_script_name} for decode sweeping")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # generic
    parser.add_argument(
        "--sbatch_args",
        default=(
            "-p devlab,learnlab,learnfair,scavenge "
            "--nodes=1 --ntasks-per-node=1 --gpus-per-node=1 "
            "--cpus-per-task=8 --mem=160gb --time=1:00:00 "
        ),
    )
    parser.add_argument("--test_modalities", nargs="+", default=["a", "v", "av"])
    parser.add_argument("--script_dir", default="./eval-scripts")
    parser.add_argument("--name_level", default=1, type=int)
    parser.add_argument("--max_tokens", default=800, type=int)

    # test sets
    parser.add_argument("--subsets", default=["valid", "test"], type=str, nargs="+")
    parser.add_argument(
        "--noise_dir",
        default="/checkpoint/wnhsu/data/avhubert_manifest/musan",
        type=str,
    )
    parser.add_argument(
        "--noise_subdirs", default=["music", "noise", "speech"], type=str, nargs="*"
    )
    parser.add_argument("--snrs", default=[0], type=int, nargs="*")

    # decode params
    parser.add_argument("--beams", default=[10], type=int, nargs="+")
    parser.add_argument("--lenpens", default=[0], type=int, nargs="+")
    parser.add_argument("--exp_dirs", nargs="+")

    args = parser.parse_args()

    batch_get_scripts(
        args.exp_dirs,
        args.script_dir,
        args.test_modalities,
        args.subsets,
        args.name_level,
        args.sbatch_args,
        args.noise_dir,
        args.noise_subdirs,
        args.snrs,
        args.beams,
        args.lenpens,
        args.max_tokens,
    )
