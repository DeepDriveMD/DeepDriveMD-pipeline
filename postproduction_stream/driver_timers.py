from radical.entk import Pipeline, Stage, Task, AppManager
import os
import glob
import argparse

if os.environ.get("RADICAL_ENTK_VERBOSE") is None:
    os.environ["RADICAL_ENTK_REPORT"] = "True"

try:
    hostname = os.environ.get("RMQ_HOSTNAME")
    port = int(os.environ.get("RMQ_PORT"))
    username = os.environ.get("RMQ_USERNAME")
    password = os.environ.get("RMQ_PASSWORD")
except Exception as e:
    print(e)


def generate_pipeline(logs, pattern, args):
    p = Pipeline()
    p.name = "Postproduction"

    ex_start, ex_end = list(map(int, args.exclude.split("-")))

    s = Stage()
    s.name = "Timers"
    for task in range(len(logs)):
        if task in range(ex_start, ex_end + 1):
            print(f"Skipping task {task}")
            continue
        t = Task()
        t.name = f"Task{task}"
        t.executable = "/usr/workspace/cv_ddmd/conda1/powerai/bin/python"
        t.arguments = [
            "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/postproduction_stream/timers1.py",
            "-s",
            args.session,
            "-p",
            args.pilot,
            "-t",
            task,
        ]
        t.pre_exec = ["source /usr/workspace/cv_ddmd/software1/etc/powerai.sh"]
        t.cpu_reqs = {
            "processes": 1,
            "threads_per_process": 4,
            "process_type": "MPI",
            "thread_type": "OpenMP",
        }
        s.add_tasks(t)

    p.add_stages(s)

    s = Stage()
    s.name = "AggregateRMSD"

    t = Task()
    t.name = "Aggregate"
    t.executable = "/usr/workspace/cv_ddmd/conda1/powerai/bin/python"
    t.arguments = [
        "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/postproduction_stream/timers1_agg.py",
        "'" + pattern + "'",
        args.output_dir,
    ]
    t.pre_exec = ["source /usr/workspace/cv_ddmd/software1/etc/powerai.sh"]
    s.add_tasks(t)

    p.add_stages(s)

    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        "-o",
        help="a number corresponding to a directory in /p/gpfs1/yakushin/Outputs",
        #type=int,
        #default=3,
    )
    parser.add_argument(
        "--nodes", "-n", help="number of nodes to use", type=int, default=1
    )
    parser.add_argument(
        "--walltime", "-w", help="walltime in minutes", type=int, default=10
    )
    parser.add_argument(
        "--session",
        "-s",
        help="re.session*",
    )
    parser.add_argument("--pilot", "-p", help="pilot", type=int, default=0)
    parser.add_argument(
        "--exclude",
        "-e",
        help="start-end - range to exclude, for example to avoid dealing with aggregator logs",
        default="120-129",
    )

    args = parser.parse_args()

    pilot = f"pilot.{args.pilot:04d}"

    pattern = (
        f"/p/gpfs1/yakushin/radical.pilot.sandbox/{args.session}/{pilot}/task.*/*.out"
    )
    logs = glob.glob(pattern)
    logs.sort()
    print("len(logs) = ", len(logs))

    # Create Application Manager
    appman = AppManager(
        hostname=hostname,
        port=port,
        username=username,
        password=password,
        reattempts=20,
        resubmit_failed=True,
    )

    # Create a dictionary describe four mandatory keys:
    # resource, walltime, and cpus
    # resource is 'local.localhost' to execute locally
    res_dict = {
        "resource": "llnl.lassen",
        "walltime": args.walltime,
        "cpus": 39 * 4 * args.nodes,
        "gpus": 0,
        "queue": "pbatch",
        "schema": "local",
        "project": "cv19-a01",
    }

    # Assign resource request description to the Application Manager
    appman.resource_desc = res_dict

    # Assign the workflow as a set or list of Pipelines to the Application Manager
    # Note: The list order is not guaranteed to be preserved
    appman.workflow = set(
        [generate_pipeline(logs, pattern.replace("*.out", "timers.csv"), args)]
    )

    # Run the Application Manager
    appman.run()
