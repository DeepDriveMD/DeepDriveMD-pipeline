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


def generate_pipeline(trajectories, pattern, nodes, compute_zcentroid):
    p = Pipeline()
    p.name = "Postproduction"

    s = Stage()
    s.name = f"TrajectoryToRMSD"
    for task in range(len(trajectories)):
        tr = trajectories[task]
        t = Task()
        t.name = f"Task{task}"
        t.executable = "/usr/workspace/cv_ddmd/conda1/powerai/bin/python"
        t.arguments = [
            "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/postproduction_stream/rmsd1.py",
            tr,
            compute_zcentroid,
        ]
        t.pre_exec = ["source /usr/workspace/cv_ddmd/software1/etc/powerai.sh"]
        s.add_tasks(t)

    p.add_stages(s)

    s = Stage()
    s.name = "AggregateRMSD"

    t = Task()
    t.name = "Aggregate"
    t.executable = "/usr/workspace/cv_ddmd/conda1/powerai/bin/python"
    t.arguments = [
        "/usr/workspace/cv_ddmd/yakushin/Integration1/DeepDriveMD-pipeline/postproduction_stream/rmsd1_agg.py",
        "'" + pattern + "'",
        compute_zcentroid,
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
        type=int,
        default=3,
    )
    parser.add_argument(
        "--nodes", "-n", help="number of nodes to use", type=int, default=1
    )
    parser.add_argument(
        "--walltime", "-w", help="walltime in minutes", type=int, default=10
    )
    parser.add_argument("--compute_zcentroid", "-z", help="1 or 0", type=int, default=0)

    args = parser.parse_args()

    pattern = f"/p/gpfs1/yakushin/Outputs/{args.output_dir}/molecular_dynamics_runs/stage0000/task*/*/trajectory.bp"
    trajectories = glob.glob(pattern)
    trajectories.sort()
    print("len(trajectories) = ", len(trajectories))

    # Create Application Manager
    appman = AppManager(
        hostname=hostname, port=port, username=username, password=password
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
        [
            generate_pipeline(
                trajectories,
                pattern.replace("trajectory.bp", "rmsd.csv"),
                args.nodes,
                args.compute_zcentroid,
            )
        ]
    )

    # Run the Application Manager
    appman.run()
