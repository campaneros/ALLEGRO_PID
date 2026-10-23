import argparse
import os
import shlex
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EOS = "/eos/user/m/mcampana/ALLEGRO/kubeflow"
# pytorch 2.5 as in LCG_107, through the CERN docker hub proxy (no pull rate limit)
IMAGE = "registry.cern.ch/docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


def main():
    # without allow_abbrev=False the --n option of train.py would be read as --namespace
    ap = argparse.ArgumentParser(description="Run train.py on the CERN Kubeflow (ml.cern.ch); every unknown option goes to "
                                 "train.py", allow_abbrev=False)
    ap.add_argument("--cpu", default="4", help="CPU limit")
    ap.add_argument("--mem", default="16G", help="memory limit")
    ap.add_argument("--namespace", default=os.environ.get("USER"), help="Kubeflow profile namespace")
    a, rest = ap.parse_known_args()  # rest: the arguments of train.py

    # copy the code to EOS, the pod reads it from there
    run = f"pid_{time.strftime('%Y%m%d_%H%M%S')}"
    code = f"{EOS}/{run}/code"
    shutil.copytree(HERE, code, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pt", "*.log", "run*", "smoke"))
    print("code copied to", code)

    # command run in the pod: a venv on top of the torch of the image, then the training
    script = (f"set -e; python -m venv --system-site-packages /tmp/venv && . /tmp/venv/bin/activate && "
              f"cd {code} && pip install -q --no-cache-dir -r requirements.txt uproot && python train.py {shlex.join(rest)}")

    from kfp import dsl, kubernetes
    from kfp.client import Client

    @dsl.container_component
    def train():
        return dsl.ContainerSpec(image=IMAGE, command=["bash", "-c", script])

    @dsl.pipeline(name="allegro-pid")
    def pipeline():
        t = train().set_cpu_limit(a.cpu).set_memory_limit(a.mem).set_caching_options(False)
        kubernetes.add_pod_label(t, label_key="mount-eos", label_value="true")
        kubernetes.use_secret_as_env(t, secret_name="comet", secret_key_to_env={"COMET_API_KEY": "COMET_API_KEY"})

    # a token is needed only when submitting from outside ml.cern.ch
    tok = os.environ.get("KF_TOKEN")
    client = Client(host="https://ml.cern.ch/pipeline", existing_token=tok) if tok else Client()
    r = client.create_run_from_pipeline_func(pipeline, run_name=f"allegro-pid {run}", experiment_name="pid-allegro",
                                             namespace=a.namespace)
    print("run", r.run_id, "- ml.cern.ch, Pipelines - Runs; metrics at comet.com, workspace campaneros, project pid-allegro")


if __name__ == "__main__":
    main()
