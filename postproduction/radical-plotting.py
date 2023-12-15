#!/usr/bin/env python

import os

import matplotlib              as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot       as plt
import matplotlib.ticker       as mticker

import pandas                  as pd
import numpy                   as np

import radical.analytics       as ra
import radical.pilot           as rp
import radical.utils           as ru

from radical.analytics.utils import to_latex

import subprocess
import argparse

subprocess.run(["rm","-rf","~/.radical/analytics/cache"])

#fm.fontManager.addfont(
#    fm.findSystemFonts(os.path.join(os.getcwd(), './fonts'))[0])

plt.style.use(ra.get_mplstyle('radical_mpl'))
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.serif']  = ['Nimbus Roman Becker No9L']
mpl.rcParams['font.family'] = 'serif'

DURATIONS = {
    'boot'        : [{ru.EVENT: 'bootstrap_0_start'},
                     {ru.EVENT: 'bootstrap_0_ok'   }],
    'agent_setup' : [{ru.EVENT: 'bootstrap_0_ok'   },
                     {ru.STATE: rp.PMGR_ACTIVE     }],
    'exec_queue'  : [{ru.STATE: rp.AGENT_SCHEDULING},
                     {ru.STATE: rp.AGENT_EXECUTING }],
    'exec_prep'   : [{ru.STATE: rp.AGENT_EXECUTING },
                     {ru.EVENT: 'task_run_start'   }],
    'exec_rp'     : [{ru.EVENT: 'task_run_start'   },
                     {ru.EVENT: 'launch_start'     }],
    'exec_sh'     : [{ru.EVENT: 'launch_start'     },
                     {ru.EVENT: 'launch_submit'    }],
    'exec_launch' : [{ru.EVENT: 'launch_submit'    },
                     {ru.EVENT: 'exec_start'       }],
    'exec_cmd'    : [{ru.EVENT: 'exec_start'       },
                     {ru.EVENT: 'exec_stop'        }],
    'exec_finish' : [{ru.EVENT: 'exec_stop'        },
                     {ru.EVENT: 'launch_collect'   }],
    'term_sh'     : [{ru.EVENT: 'launch_collect'   },
                     {ru.EVENT: 'launch_stop'      }],
    'term_rp'     : [{ru.EVENT: 'launch_stop'      },
                     {ru.EVENT: 'task_run_stop'    }],
    'unschedule'  : [{ru.EVENT: 'task_run_stop'    },
                     {ru.EVENT: 'unschedule_stop'  }]
}

# configurable - set necessary time ranges under a processing state\n",

METRICS = [
    ['Bootstrap', ['boot', 'agent_setup'],      '#c6dbef'],
    ['Schedule',  ['exec_queue', 'unschedule'], '#c994c7'],
    ['Prep',      ['exec_prep', 'exec_rp', 'exec_sh',
                   'term_sh', 'term_rp'],       '#fdbb84'],
    ['Launch',    ['exec_launch'],              '#ff9999'],
    ['Running',   ['exec_cmd'],                 '#88bb88']
]

def init_session(sid):
    """
    Create a session based on the session ID (sid) passed in
    """
    session = ra.Session(sid, 'radical.pilot')

    data = {'session': session,
            'pilot'  : session.filter(etype='pilot', inplace=False).get()[0],
            'tasks'  : session.filter(etype='task',  inplace=False)}

    data['sid'] = sid
    data['pid'] = data['pilot'].uid
    data['smt'] = (os.environ.get('RADICAL_SMT', 1) or
                   data['pilot'].cfg['resource_details']['rm_info']['threads_per_core'])
    return data

def print_metrics(data):
    ttx = data['tasks'].duration(event=[{ru.EVENT: 'task_run_start'},
                                        {ru.EVENT: 'task_run_stop'}])
    runtime = data['pilot'].duration(event=[{ru.EVENT: 'bootstrap_0_start'},
                                            {ru.EVENT: 'bootstrap_0_stop'}])
    print('OVH: %s sec | Total time: %s sec' % (round(runtime - ttx), round(runtime)))

    # calculate scheduling throughput
    ts_schedule_ok = sorted(session.timestamps(event={ru.STATE: 'AGENT_SCHEDULING'}))
    print('scheduling throughput: ', round(len(ts_schedule_ok) / (ts_schedule_ok[-1] - ts_schedule_ok[0]), 3))

    # calculate launching rate
    ts_agent_executing = sorted(session.timestamps(event=[{ru.EVENT: 'launch_submit'}]))
    print('launching rate: ', round(len(ts_agent_executing) / (ts_agent_executing[-1] - ts_agent_executing[0]), 3))

def plot_data1(data):
    sid = data['sid']
    pid = data['pid']

    rtype_info = {'cpu': {'label': 'Number of CPU cores',
                          'formatter': lambda z, pos: int(z / data['smt'])},
                  'gpu': {'label': 'Number of GPUs',
                          'formatter': None}}
    
    exp = ra.Experiment([sid], stype='radical.pilot')
    
    correction = 0.5
    
    # get the start time of each pilot
    p_zeros = ra.get_pilots_zeros(exp)
    
    rtypes = ['cpu', 'gpu']
    # fig, axarr = plt.subplots(1, len(rtypes), figsize=(
    #     ra.get_plotsize(256 * len(rtypes), subplots=(1, len(rtypes)))))
    
    fig, axarr = plt.subplots(len(rtypes), 1, figsize=(7, 2 * len(rtypes)))
    
    sub_label = 'a'
    legend = None
    for idx, rtype in enumerate(rtypes):
    
        if len(rtypes) > 1:
            ax = axarr[idx]
        else:
            ax = axarr
    
        consumed = rp.utils.get_consumed_resources(
            exp._sessions[0], rtype, {'consume': DURATIONS})
    
        # generate the subplot with labels
        legend, patches, x, y = ra.get_plot_utilization(
            METRICS, {sid: consumed}, p_zeros[sid][pid], sid)
    
        # place all the patches, one for each metric, on the axes
        for patch in patches:
            patch.set_y(patch.get_y() + correction)
            ax.add_patch(patch)
    
        ax.set_xlim([x['min'], int(x['max'])])
        ax.set_ylim([y['min'] + correction, int(y['max'] + correction)])
    
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    
        if rtype_info[rtype]['formatter'] is not None:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                rtype_info[rtype]['formatter']))
    
        if len(rtypes) > 1:
            ax.set_xlabel('(%s)' % sub_label, labelpad=10)
        ax.set_ylabel(to_latex(rtype_info[rtype]['label']), fontsize=11)
        ax.set_title(' ')  # placeholder
    
        sub_label = chr(ord(sub_label) + 1)
    
    fig.legend(legend, [m[0] for m in METRICS],
               loc='upper center',
               bbox_to_anchor=(0.5, 1.03),
               ncol=len(METRICS))
    fig.text(0.5, 0.05, 'Time (s)', ha='center')
    
    #plt.tight_layout()
    plt.show()
    
    plot_name = '%s.ru.png' % '.'.join(data['sid'].rsplit('.', 2)[1:])
    fig.savefig(plot_name)

def plot_data2(data):
    tmap = {
        'pilot':  [
            [{1: 'bootstrap_0_start'}     , 'system'     , 'Bootstrap'  ],
            [{5: 'PMGR_ACTIVE'}           , 'Bootstrap'  , 'Idle'       ],
            [{1: 'cmd', 6: 'cancel_pilot'}, 'Idle'       , 'Term'       ],
            [{1: 'bootstrap_0_stop'}      , 'Term'       , 'system'     ],
            [{1: 'sub_agent_start'}       , 'Idle'       , 'agent'      ],
            [{1: 'sub_agent_stop'}        , 'agent'      , 'Term'       ]
        ],
        'task': [
            [{1: 'schedule_ok'}           , 'Idle'       , 'Exec setup' ],
            [{1: 'exec_start'}            , 'Exec setup' , 'Running'    ],
            [{1: 'exec_stop'}             , 'Running'    , 'Exec setup' ],
            [{1: 'unschedule_stop'}       , 'Exec setup' , 'Idle'       ]
        ],
    }
    metrics = [
        # metric,      line color, alpha, fill color, alpha
        ['Bootstrap',  ['#c6dbef', 0.0, '#c6dbef', 1]],
        ['Exec setup', ['#fdbb84', 0.0, '#fdbb84', 1]],
        ['Running',    ['#88bb88', 0.0, '#88bb88', 1]],
        ['Idle',       ['#f0f0f0', 0.0, '#f0f0f0', 1]]
    ]
    
    to_stack = [m[0] for m in metrics]
    to_plot = {m[0]: m[1] for m in metrics}
    
    rtypes = ['cpu', 'gpu']
    fig, axarr = plt.subplots(len(rtypes), 1, figsize=(7, 2 * len(rtypes)))
    
    patches = []
    legend = []
    
    sub_label = 'a'
    
    sid = data['sid']
    pid = data['pid']
    
    p = data['pilot']
    rm_info = p.cfg['resource_details']['rm_info']
    p_size  = p.description['cores']
    n_nodes = int(p_size / rm_info['cores_per_node'])
    n_tasks = len(data['tasks'].get())
    p_resrc, series, x = ra.get_pilot_series(
        data['session'], p, tmap, ['cpu', 'gpu'], True)
    
    for idx, rtype in enumerate(rtypes):
    
        if len(rtypes) > 1:
            ax = axarr[idx]
        else:
            ax = axarr
    
        # stack timeseries for each metrics into areas
        areas = ra.stack_transitions(series, rtype, to_stack)
    
        # plot individual metrics
        prev_m = None
        for m in areas:
    
            if m not in to_plot:
                if m != 'time':
                    print('skip', m)
                continue
    
            lcol = to_plot[m][0]
            lalpha = to_plot[m][1]
            pcol = to_plot[m][2]
            palpha = to_plot[m][3]
    
            # plot the (stacked) areas
            ax.step(np.array(areas['time']), np.array(areas[m]),
                    where='post', label=m,
                    color=lcol, alpha=lalpha, linewidth=1.0)
    
            # fill first metric toward 0, all others towards previous line
            if not prev_m:
                patch = ax.fill_between(
                    areas['time'], areas[m],
                    step='post', label=m,
                    linewidth=0.0,
                    color=pcol, alpha=palpha)
    
            else:
                patch = ax.fill_between(
                    areas['time'], areas[m],
                    areas[prev_m],
                    step='post', label=m,
                    linewidth=0.0,
                    color=pcol, alpha=palpha)
    
            # remember patches for legend
            if len(legend) < len(metrics):
                legend.append(m)
                patches.append(patch)
    
            # remember this line to fill against
            prev_m = m
    
        ax.set_xlim([x['min'], x['max']])
        ax.set_ylim([0, 110])
    
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(3, steps=[5, 10]))
    
        ax.set_ylabel('%s (%%)' % rtype.upper(), fontsize=12)
    
        for ax in fig.get_axes():
            ax.label_outer()
    
    fig.legend(
        patches, legend, loc='upper center', ncol=6,
        bbox_to_anchor=(0.5, 1.035),
        fancybox=True, shadow=True, fontsize=11)
    fig.text(0.5, 0.008, 'Time (s)', ha='center', size=11)
    
    plt.tight_layout()
    plt.show()
    
    plot_name = '%s.ru.stack.png' % '.'.join(data['sid'].rsplit('.', 2)[1:])
    fig.savefig(os.path.join('.', plot_name))

def plot_data3(data):
    PAGE_WIDTH = 506
    
    events = [('Tasks scheduling', [{ru.STATE: 'AGENT_SCHEDULING'},
                                    {ru.EVENT: 'schedule_ok'}]),
              ('Tasks running',    [{ru.EVENT: 'exec_start'},
                                    {ru.EVENT: 'exec_stop'}])]
    
    fig, ax = plt.subplots(1,1, figsize=(7, 2))
    
    pilot_starttime = data['pilot'].\
        timestamps(event={ru.EVENT: 'bootstrap_0_start'})[0]
    
    for e_name, e_range in events:
        time_series = data['session'].concurrency(event=e_range, sampling=1)
        ax.plot([e[0] - pilot_starttime for e in time_series],
                [e[1] for e in time_series],
                label=ra.to_latex(e_name), lw=1)
    
    fig.legend([e[0] for e in events],
               loc='upper center',
               bbox_to_anchor=(0.75, 1.0),
               ncol=2, fontsize=10)
    ax.set_ylabel(to_latex('Number of tasks'), fontsize=11)
    fig.text(0.5, 0.008, 'Time (s)', ha='center', size=11)
    
    plt.tight_layout()
    plt.show()
    plot_name = '%s.concurrency.png' % '.'.join(data['sid'].rsplit('.', 2)[1:])
    fig.savefig(os.path.join('.', plot_name))

def parse_args():
    """
    Get the session id from the command line arguments.
    """
    describe = '''
Plot the RADICAL-Cybertool performance data.'''
    parser = argparse.ArgumentParser(description=describe,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sid",help="The session name of the workflow run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = init_session(args.sid)
    try: print_metrics(data)
    except: pass
    plot_data1(data)
    plot_data2(data)
    plot_data3(data)
